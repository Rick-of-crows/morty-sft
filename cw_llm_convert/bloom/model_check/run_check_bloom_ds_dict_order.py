import torch
import os
import json
import argparse

# from megatron import get_args
# from megatron import print_rank_0

# from megatron import mpu
# from megatron.enums import AttnMaskType
# from megatron.model.gpt_model import GPTModelPipe
# from megatron.initialize import initialize_megatron
# from megatron.checkpointing import save_checkpoint

# import deepspeed
# from deepspeed.runtime.utils import see_memory_usage
# import subprocess

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

    ckpt_list=[i for i in os.listdir(path_hf) if 'layer_' in i]
    assert ckpt_list != [], 'can not find any ckpt, ckpt_list is empty'
    ckpt_list.sort()

    ds_state_1 = []
    for idx, ckpt in enumerate(ckpt_list):
        # import pdb; pdb.set_trace()
        bloom_checkpoint = torch.load(os.path.join(path_hf, ckpt), map_location='cpu')
        print("checkpoint: {} Loaded".format(os.path.join(path_hf, ckpt)))
        for name in bloom_checkpoint.keys():
            ds_state_1.append(name)
   

    ckpt_list = [i for i in os.listdir(path_ds) if 'layer_' in i and '00-model_states.pt' in i]
    assert ckpt_list != [], 'can not find any ckpt, ckpt_list is empty'
    ckpt_list.sort()
    ds_state_2 = []
    for idx, ckpt in enumerate(ckpt_list):
        # import pdb; pdb.set_trace()
        bloom_checkpoint = torch.load(os.path.join(path_ds, ckpt), map_location='cpu')
        print("checkpoint: {} Loaded".format(os.path.join(path_ds, ckpt)))
        for name in bloom_checkpoint.keys():
            ds_state_2.append(name)


    for idx, name1 in enumerate(ds_state_1):
        Verified = name1 == ds_state_2[idx]
        print("Verified: {} for layer: {}".format(Verified, name1))


if __name__=='__main__':

    
    bloom_path1 = '/workspace/bloomz-mt-sd/global_step0/'
    bloom_path2 = '/data/output_dir/2023-05-31/bloomz-176-cw-sft-pretrain/global_step2'

    convert_bloom_to_deepspeed(bloom_path1, bloom_path2)


