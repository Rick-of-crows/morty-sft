import torch
import os
import json
import argparse

from megatron import get_args
from megatron import print_rank_0

from megatron import mpu
from megatron.enums import AttnMaskType
from megatron.model.gpt_model import GPTModelPipe
from megatron.initialize import initialize_megatron
from megatron.checkpointing import save_checkpoint

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
import subprocess

from megatron.training import setup_model_and_optimizer

from megatron.model.module import fp32_to_float16

def get_bloom_ckpt(path):

    # ckpt_list=['pytorch_model-00001-of-00033.bin']
    # ckpt_list=['pytorch_model_00001-of-00032.bin']

    ckpt_list=[i for i in os.listdir(path) if 'pytorch_model-' in i]
    #ckpt_list=[i for i in os.listdir(path) if 'pytorch_model_' in i]
    assert ckpt_list != [], 'can not find any ckpt, ckpt_list is empty'
    ckpt_list.sort()
    bloom_checkpoint =[]
    
    layer_list={}
    # emb_layer=[]
    for idx, ckpt in enumerate(ckpt_list):
        # import pdb; pdb.set_trace()
        # llama_checkpoint.append(torch.load(os.path.join(path, ckpt), map_location='cpu'))
        bloom_checkpoint=torch.load(os.path.join(path, ckpt), map_location='cpu')
        # layer = [(key, value.size()) for key, value in llama_checkpoint[idx].items()]
        # for id, l in enumerate(layer):
        #     print(f"{id}: {l}")
        
        for name, param in bloom_checkpoint.items():
            # todo: 默认全部转成 fp16 - 支持7b
            if param.dtype == torch.float32:
                param = fp32_to_float16(param, lambda v: v.half())
            # layer_list[f"{name}"]=param

            # # nemo has `transformer` prefix, remove, hf source not
            name = '.'.join(name.split('.')[1:])
            layer_list[f"{name}"]=param
        
        # emb_layer.extend([(key, value) for key, value in llama_checkpoint[idx].items() if 'emb' in key])
        
    layer_dic={}
    for k in layer_list.keys():
        layer_dic[f"{k}"]=0
    
    # import pdb; pdb.set_trace()
    # model=AutoModelForCausalLM.from_pretrained("checkpoints/llama-7b")
    # model=LlamaForCausalLM.from_pretrained("checkpoints/llama-7b")
    # layer=[(name, para.size()) for name, para in model.named_parameters()]
    # for _, (name, para_size) in enumerate(layer):
    #     print(f"{name}: {para_size}")
    #     # count parameter: 6738415616;  log: 6739197952; 700000 more in log (same as bloom)
    #     if len(para_size) == 2:
    #         count += para_size[0] * para_size[1]
    #     else:
    #         count += para_size[0]
    
    return bloom_checkpoint, layer_list, layer_dic

# def build_deepspeed_model():
#     """Build the model."""

#     initialize_megatron()
    
#     print_rank_0('building GPT model ...')
#     see_memory_usage(f"Before Building Model", force=True)

#     args = get_args()

#     with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
#                              remote_device=None if args.remote_device == 'none' else args.remote_device,
#                              config_dict_or_path=args.deepspeed_config,
#                              enabled=args.zero_stage == 3,
#                              mpu=mpu):
#         if args.deepspeed:
#             model = LlamaModelPipe(
#                 num_tokentypes=0,
#                 parallel_output=True,
#                 attn_mask_type=AttnMaskType.prefix
#             )
    
#     see_memory_usage(f"After Building Model", force=True)
#     return model


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    see_memory_usage(f"Before Building Model", force=True)

    args = get_args()

    with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=args.deepspeed_config,
                             enabled=args.zero_stage == 3,
                             mpu=mpu):
        assert args.deepspeed==True, 'deepspeed need to be set to True'
        if args.deepspeed:
            model = GPTModelPipe(
                num_tokentypes=0,
                parallel_output=True,
                attn_mask_type=AttnMaskType.prefix
            )
            # import pdb; pdb.set_trace()
            # This is a hack to give us a reference to get_batch_pipe from within training.py
            # We need to call model.set_batch_fn after deepspeed.initialize
            model._megatron_batch_fn = get_batch_pipe

        else:
            model = GPTModelPipe(
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process,
                prefix_lm=True
            )
    see_memory_usage(f"After Building Model", force=True)
    return model

def get_batch_pipe(data):
    """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
    args = get_args()

    return

def input_layernorm(x):
    layer = f"{int(x.split('.')[1]) - 3}."
    return "h." + layer + '.'.join(x.split('.')[2:])
def attn_qkv(x):
    layer = f"{int(x.split('.')[1]) - 3}."
    return "h." + layer + '.'.join(x.split('.')[2:])
def attn_dense(x):
    layer = f"{int(x.split('.')[1]) - 3}."
    return "h." + layer + '.'.join(x.split('.')[2:])
def word_embedding(x):
    return "word_embeddings.weight"
def word_embedding_norm_weight(x):
    return "word_embeddings_layernorm.weight"
def word_embedding_norm_bias(x):
    return "word_embeddings_layernorm.bias"
def layernorm(x):
    return "model.norm.weight"
def final_layernorm_weight(x):
    return "ln_f.weight"
def final_layernorm_bias(x):
    return "ln_f.bias"

def layer_mapping(x):
    dic_mapping = {
        "input_layernorm.weight": input_layernorm,
        "input_layernorm.bias": input_layernorm,
        "self_attention.query_key_value.weight": attn_qkv,
        "self_attention.query_key_value.bias": attn_qkv,
        "self_attention.dense.weight": attn_dense,
        "self_attention.dense.bias": attn_dense,
        "mlp.dense_4h_to_h.weight": input_layernorm,
        "mlp.dense_4h_to_h.bias": input_layernorm,
        "mlp.dense_h_to_4h.weight": input_layernorm,
        "mlp.dense_h_to_4h.bias": input_layernorm,
        "post_attention_layernorm.weight": input_layernorm,
        "post_attention_layernorm.bias": input_layernorm,
        "embed.word_embeddings.weight": word_embedding,
        "embed.word_embeddings.norm.weight": word_embedding_norm_weight,
        "embed.word_embeddings.norm.bias": word_embedding_norm_bias,
        "weight": final_layernorm_weight,
        "bias": final_layernorm_bias,
    }
    return dic_mapping[f"{'.'.join(x.split('.')[2:])}"](x)


def convert_bloom_to_deepspeed(layer_list, model, layer_dict):
    
    for i, (name, para) in enumerate(model[0].named_parameters()):
        #import pdb; pdb.set_trace()
        hf_layer=layer_mapping(name)
        print(f"Converting {name} to {hf_layer} ...") 
        para.data.copy_(layer_list[f"{hf_layer}"])
        layer_dict[f"{hf_layer}"]=1
        print(f"Done convert {name} to {hf_layer}")
        
        # only skip `rotary_emb.inv_freq` weight, all is same:
        # tensor([1.0000e+00, 8.6596e-01, 7.4989e-01, 6.4938e-01, 5.6234e-01, 4.8697e-01, ...
    
    # can use layer_dict.values() to check if any hugging face checkpoint is not used
    # import pdb; pdb.set_trace()
    return model


if __name__=='__main__': 
    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron()
    args = get_args()

    # path='checkpoints/bloomz-7b1-hf'
    assert os.path.exists(args.bloom_hf_ckpt)==True, 'bloom hugging face checkpoint path not exists'
    bloom_ckpt_list, layer_list, layer_dict=get_bloom_ckpt(args.bloom_hf_ckpt)
    
    if args.deepspeed:
        args.deepspeed_configuration = json.load(
            open(args.deepspeed_config, 'r', encoding='utf-8'))
        
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider)
    
    model=convert_bloom_to_deepspeed(layer_list, model, layer_dict)
    
    torch.distributed.barrier()
    save_checkpoint(0, model, optimizer, lr_scheduler)
    torch.distributed.barrier()
    
    
    
