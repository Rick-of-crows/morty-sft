import torch
import os
import json
import argparse

from megatron import get_args
from megatron import print_rank_0

from megatron import mpu
from megatron.enums import AttnMaskType
from megatron.model.gpt_model_llama import LlamaModelPipe
from megatron.initialize import initialize_megatron
from megatron.checkpointing import save_checkpoint

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
import subprocess

from megatron.training import setup_model_and_optimizer



# from transformers import LlamaPreTrainedModel, LlamaForCausalLM
# from transformers import AutoModelForCausalLM

def get_llama_ckpt(path):

    # ckpt_list=['pytorch_model-00001-of-00033.bin']
    # ckpt_list=['pytorch_model_00001-of-00032.bin']
    
    ckpt_list=[i for i in os.listdir(path) if 'pytorch_model-' in i]
    ckpt_list.sort()
    llama_checkpoint =[]
    
    layer_list={}
    # emb_layer=[]
    for idx, ckpt in enumerate(ckpt_list):
        # import pdb; pdb.set_trace()
        # llama_checkpoint.append(torch.load(os.path.join(path, ckpt), map_location='cpu'))
        llama_checkpoint=torch.load(os.path.join(path, ckpt), map_location='cpu')
        # layer = [(key, value.size()) for key, value in llama_checkpoint[idx].items()]
        # for id, l in enumerate(layer):
        #     print(f"{id}: {l}")
        
        for name, param in llama_checkpoint.items():
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
    
    return llama_checkpoint, layer_list, layer_dic

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
            model = LlamaModelPipe(
                num_tokentypes=0,
                parallel_output=True,
                attn_mask_type=AttnMaskType.prefix
            )
            # import pdb; pdb.set_trace()
            # This is a hack to give us a reference to get_batch_pipe from within training.py
            # We need to call model.set_batch_fn after deepspeed.initialize
            model._megatron_batch_fn = get_batch_pipe

        else:
            model = LlamaModelPipe(
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
    return "model.layers." + layer + '.'.join(x.split('.')[2:])
def attn_qkv(x):
    layer = f"{int(x.split('.')[1]) - 3}."
    return ["model.layers." + layer + f"self_attn.{i}_proj.weight" for i in ["q", "k", "v"]]
def attn_o(x):
    layer = f"{int(x.split('.')[1]) - 3}."
    return "model.layers." + layer + f"self_attn.o_proj.weight"
def word_embedding(x):
    return "model.embed_tokens.weight"
def layernorm(x):
    return "model.norm.weight"
def lm_head(x):
    return "lm_head.weight"
def final_linear(x):
    return "lm_head.weight"
def layer_mapping(x):
    dic_mapping = {
        "input_layernorm.weight": input_layernorm,
        "self_attention.query_key_value.weight": attn_qkv,
        "self_attention.dense.weight": attn_o,
        "mlp.up_proj.weight": input_layernorm,
        "mlp.gate_proj.weight": input_layernorm,
        "mlp.down_proj.weight": input_layernorm,
        "post_attention_layernorm.weight": input_layernorm,
        "embed.word_embeddings.weight": word_embedding,
        "weight": layernorm,
        "lm_head.word_embeddings.weight": lm_head,
        "final_linear.weight": final_linear,
    }
    return dic_mapping[f"{'.'.join(x.split('.')[2:])}"](x)


def convert_llama_to_deepspeed(layer_list, model, layer_dict):
    
    for i, (name, para) in enumerate(model[0].named_parameters()):
        hf_layer=layer_mapping(name)
        print(f"Converting {name} to {hf_layer} ...")
        if "query_key_value" in name:
            # para.data.copy_(torch.cat((layer_list[f"{hf_layer[0]}"], layer_list[f"{hf_layer[1]}"], layer_list[f"{hf_layer[2]}"]), dim=0))
            # layer_dict[f"{hf_layer[0]}"], layer_dict[f"{hf_layer[1]}"], layer_dict[f"{hf_layer[2]}"]=1, 1, 1
            
            # 原来认为：张量的切分是 [q; k; v]，但是并不是 - 20230530
            # 根据 transformer_llama 中 `mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)` 的操作，合理的切分应该是：
            # [q[:head_hidden]; k[head_hidden:2*head_hidden]; v[2*head_hidden:3*head_hidden]; q[3*head_hidden:4*head_hidden]; ...]
            # 不过因为没经过训练，因此并没有影响到 hf -> ds -> hf 的推理结果
            query, key, value = layer_list[hf_layer[0]], layer_list[hf_layer[1]], layer_list[hf_layer[2]]
            qkv = torch.zeros((3 * query.shape[0], query.shape[1]))
            assert query.shape[1] == args.hidden_size, "Dimension mismatch: hidden size in hyper-parameter is {}; \
                hidden size in checkpoint is {}".format(args.hidden_size, query.shape[1])
            
            per_head_hidden_size = int(args.hidden_size / args.num_attention_heads)
            offset = 0
            offsetqkv = 0
            for _ in range(args.num_attention_heads):
                qkv[offsetqkv:offsetqkv + per_head_hidden_size] = query[offset:offset + per_head_hidden_size]
                qkv[offsetqkv + per_head_hidden_size:offsetqkv + 2 * per_head_hidden_size] = key[offset:offset + per_head_hidden_size]
                qkv[offsetqkv + 2 * per_head_hidden_size:offsetqkv + 3 * per_head_hidden_size] = value[offset:offset + per_head_hidden_size]
                offset += per_head_hidden_size
                offsetqkv += 3 * per_head_hidden_size 
            
            para.data.copy_(qkv)            
            layer_dict[f"{hf_layer[0]}"], layer_dict[f"{hf_layer[1]}"], layer_dict[f"{hf_layer[2]}"]=1, 1, 1
        else:    
            para.data.copy_(layer_list[f"{hf_layer}"])
            layer_dict[f"{hf_layer}"]=1    
        print(f"Done convert {name} to {hf_layer}")
        
        # only skip `rotary_emb.inv_freq` weight, all is same:
        # tensor([1.0000e+00, 8.6596e-01, 7.4989e-01, 6.4938e-01, 5.6234e-01, 4.8697e-01, ...
        
    # import pdb; pdb.set_trace()
    return model


if __name__=='__main__': 
    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron()
    args = get_args()
    
     # path='checkpoints/llama-7b'
    # path='checkpoints/bloomz-7b1-hf'
    assert os.path.exists(args.llama_hf_ckpt)==True, 'llama hugging face checkpoint path not exists'
    llama_ckpt_list, layer_list, layer_dict=get_llama_ckpt(args.llama_hf_ckpt)
    
    if args.deepspeed:
        args.deepspeed_configuration = json.load(
            open(args.deepspeed_config, 'r', encoding='utf-8'))
        
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider)
    
    model=convert_llama_to_deepspeed(layer_list, model, layer_dict)
    
    torch.distributed.barrier()
    save_checkpoint(0, model, optimizer, lr_scheduler)
    torch.distributed.barrier()
    
    
    