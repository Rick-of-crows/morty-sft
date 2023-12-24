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

from collections import defaultdict, OrderedDict, deque

import re



def get_batch_pipe(data):
    """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
    args = get_args()

    return

def input_layernorm(x):
    return '.'.join(x.split('.')[2:])
def word_embedding(x):
    return "word_embeddings.weight"
def word_embedding_norm_weight(x):
    return "word_embeddings.norm.weight"
def word_embedding_norm_bias(x):
    return "word_embeddings.norm.bias"
def final_layernorm_weight(x):
    return "weight"
def final_layernorm_bias(x):
    return "bias"

def layer_mapping(x):
    dic_mapping = {
        "input_layernorm.weight": input_layernorm,
        "input_layernorm.bias": input_layernorm,
        "self_attention.query_key_value.weight": input_layernorm,
        "self_attention.query_key_value.bias": input_layernorm,
        "self_attention.dense.weight": input_layernorm,
        "self_attention.dense.bias": input_layernorm,
        "mlp.dense_4h_to_h.weight": input_layernorm,
        "mlp.dense_4h_to_h.bias": input_layernorm,
        "mlp.dense_h_to_4h.weight": input_layernorm,
        "mlp.dense_h_to_4h.bias": input_layernorm,
        "post_attention_layernorm.weight": input_layernorm,
        "post_attention_layernorm.bias": input_layernorm,
        "word_embeddings.weight": word_embedding,
        "word_embeddings_layernorm.weight": word_embedding_norm_weight,
        "word_embeddings_layernorm.bias": word_embedding_norm_bias,
        "ln_f.weight": final_layernorm_weight,
        "ln_f.bias": final_layernorm_bias,
    }
    # return dic_mapping[f"{'.'.join(x.split('.')[2:])}"](x)

    embed_prefix_pattern = 'word_embedding'
    ln_prefix_pattern = 'ln_f'
    is_embed = re.search(embed_prefix_pattern, x) is not None
    is_ln = re.search(ln_prefix_pattern, x) is not None
    if is_embed:
        key = x
    elif is_ln:
        key = x
    else:
        key = '.'.join(x.split('.')[2:])

    return dic_mapping[key](x)


def convert_bloom_to_deepspeed(path_hf, path_ds):

    ckpt_list=[i for i in os.listdir(path_hf) if 'pytorch_model-' in i]
    assert ckpt_list != [], 'can not find any ckpt, ckpt_list is empty'
    ckpt_list.sort()

    prefix_pattern = r'\d'

    hf_state = {}
    for idx, ckpt in enumerate(ckpt_list):
        # import pdb; pdb.set_trace()
        bloom_checkpoint = torch.load(os.path.join(path_hf, ckpt), map_location='cpu')
        for hf_name, para in bloom_checkpoint.items():
            print(hf_name)
            hf_state[hf_name] = para
            # if re.search("lm_head", hf_name):
            #     hf_name = ".".join(hf_name.split(".")[:])
            #     continue
            # hf_name = ".".join(hf_name.split(".")[1:])
            # embed_prefix_pattern = 'word_embedding'
            # ln_prefix_pattern = 'ln_f'
            # is_embed = re.search(embed_prefix_pattern, hf_name) is not None
            # is_ln = re.search(ln_prefix_pattern, hf_name) is not None
            # if is_embed:
            #     layer = 'word_embedding'
            # elif is_ln:
            #     layer = 'ln'
            # else:
            #     layer = hf_name.split('.')[1]
            #     layer_idx = int(layer)
            # try:
            #     ds_name = layer_mapping(hf_name)
            # except:
            #     print("Error mapping..." + layer)
            # if layer in ds_state.keys():
            #     ds_state[layer][ds_name] = para
            # else:
            #     ds_state[layer] = OrderedDict()
            #     ds_state[layer][ds_name] = para

    # print(ds_state.keys())
    
    print(hf_state.keys())
    print(len(hf_state.keys()))
    
    ckpt_list = [i for i in os.listdir(path_ds) if 'layer_' in i]
    assert ckpt_list != [], 'can not find any ckpt, ckpt_list is empty'
    ckpt_list.sort()
    
    ds_state = {}
    for idx, ckpt in enumerate(ckpt_list):
        layer = ckpt.split('-')[0].split('_')[1]
        bloom_checkpoint = torch.load(os.path.join(path_ds, ckpt), map_location='cpu')
        ds_layer = {}
        for name, para in bloom_checkpoint.items():
            # print(name)
            ds_layer[name] = para
        ds_state[layer] = ds_layer
    
    print(ds_state.keys())
    print(len(ds_state.keys()))
    
    for hf_name, hf_weight in hf_state.items():

        
        if re.search("lm_head", hf_name):
            # hf_name = ".".join(hf_name.split(".")[:])
            continue
        hf_name = ".".join(hf_name.split(".")[1:])
        embed_prefix_pattern = 'word_embedding'
        ln_prefix_pattern = 'ln_f'
        is_embed = re.search(embed_prefix_pattern, hf_name) is not None
        is_ln = re.search(ln_prefix_pattern, hf_name) is not None
        if is_embed:
            layer = '01'
        elif is_ln:
            layer = str(args.num_layers + 4)
        else:
            layer = int(hf_name.split('.')[1]) + 3
            layer = str(layer)
            if len(layer) == 1:
                layer = '0' + layer
            
        ds_name = layer_mapping(hf_name)
        ds_weight = ds_state[layer][ds_name]
        
        verified = torch.equal(ds_weight, hf_weight)
        print("Verified: {}, deepspeed - {} : {}, huggingface: {}".format(verified, layer, ds_name, hf_name))
        

    # for layer, sd in ds_state.items():
    #     if layer == 'word_embedding':
    #         layer_id = '01'
    #     elif layer == 'ln':
    #         layer_id = str(args.num_layers + 4)
    #     else:
    #         layer_id = str(int(layer) + 3)
    #         layer_id = (2 - len(layer_id)) * "0" + layer_id
    #     if not os.path.exists(os.path.join(path_ds, 'global_step0')):
    #         os.makedirs(os.path.join(path_ds, 'global_step0'))
    #     ckpt_name = f'global_step0/layer_{layer_id}-model_00-model_states.pt'
    #     torch.save(sd, os.path.join(path_ds, ckpt_name))

    # with open(os.path.join(path_ds, 'latest'), 'w', encoding='utf-8') as fw:
    #     fw.write('global_step0')

    return


if __name__=='__main__':
    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron()
    args = get_args()

    # path='checkpoints/bloomz-7b1-hf'
    assert os.path.exists(args.bloom_hf_ckpt)==True, 'bloom hugging face checkpoint path not exists'
    # bloom_ckpt_list, layer_list, layer_dict=get_bloom_ckpt(args.bloom_hf_ckpt)

    # if args.deepspeed:
    #     args.deepspeed_configuration = json.load(
    #         open(args.deepspeed_config, 'r', encoding='utf-8'))

    # model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider)

    # model=convert_bloom_to_deepspeed(layer_list, model, layer_dict)

    convert_bloom_to_deepspeed(args.bloom_hf_ckpt, args.save)

    # torch.distributed.barrier()
    # save_checkpoint(0, model, optimizer, lr_scheduler)
    # torch.distributed.barrier()

