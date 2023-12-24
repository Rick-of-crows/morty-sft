import argparse
import json
import re, os
from functools import partial
from multiprocessing import Pool
from typing import List, Optional, Dict

import torch
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--opt_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the transformers OPT checkpoint path.",
    )
    parser.add_argument(
        "--opt_sharded_index_path",
        default=None,
        type=str,
        required=True,
        help="Path to the transformers OPT checkpoint metadata path.",
    )
    parser.add_argument(
        "--megatron_dump_folder_path", default=None, type=str, required=True,
        help="Path to the output Megatron-DS model."
    )
    parser.add_argument(
        "--num-proc", default=1, type=int,
    )
    return parser.parse_args()


def compute_meg_ds_weight_names(num_layers: int):
    return {
        "layer_01-model_00-model_states.pt": [
            "word_embeddings.weight",
        ],
        **{
            f"layer_{str(layer_id).zfill(2)}-model_00-model_states.pt": [
                "input_layernorm.weight",
                "self_attention.query_key_value.weight",
                "self_attention.dense.weight",
                "post_attention_layernorm.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
                "self_attention.rotary_emb.inv_freq",
            ]
            for layer_id in range(3, num_layers + 3)
        },
        f"layer_{str(num_layers + 4).zfill(2)}-model_00-model_states.pt": [
            "weight",
        ],
        f"layer_{str(num_layers + 5).zfill(2)}-model_00-model_states.pt": [
            "final_linear.weight",
        ]
    }

NON_TRANSFORMERS_BLOCK_WEIGHTS = {
    "word_embeddings.weight": "model.embed_tokens.weight",
    "weight": "model.norm.weight",
    "final_linear.weight": "lm_head.weight"
}
TRANSFORMERS_BLOCK_WEIGHTS = {
    "input_layernorm.weight": ["input_layernorm.weight"],
    "self_attention.query_key_value.weight": ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"],
    "self_attention.dense.weight": ["self_attn.o_proj.weight"],
    "post_attention_layernorm.weight": ["post_attention_layernorm.weight"],
    "mlp.gate_proj.weight": ["mlp.gate_proj.weight"],
    "mlp.up_proj.weight": ["mlp.up_proj.weight"],
    "mlp.down_proj.weight": ["mlp.down_proj.weight"],
    "mlp.down_proj.weight": ["mlp.down_proj.weight"],
    "self_attention.rotary_emb.inv_freq": ["self_attn.rotary_emb.inv_freq"],
}
def get_transformers_weight_names(meg_ds_weight: str, layer_id: Optional[int]) -> List[str]:
    if layer_id is None:
        return [NON_TRANSFORMERS_BLOCK_WEIGHTS[meg_ds_weight]]
    else:
        return [f"model.layers.{layer_id}.{tfrs_block_name}" for tfrs_block_name in TRANSFORMERS_BLOCK_WEIGHTS[meg_ds_weight]]

def get_layer_id(meg_ds_filename: str, total_num_layers: int) -> Optional[int]:
    layer_id = int(re.match(r"layer_(\d*)-model_00-model_states.pt", meg_ds_filename)[1]) - 3

    if layer_id < 0:
        return None

    if layer_id >= total_num_layers:
        return None

    return layer_id

def merge_layers(layers, num_heads: int, num_key_value_heads:int, hidden_size: int):
    if len(layers) == 1:
        return layers[0].to(torch.bfloat16)
    else:
        # We merge QKV
        if len(layers[0].shape) == 1:
            # bias
            return torch.reshape(
                torch.cat(
                    [
                        layer.view(num_heads, 1, hidden_size // num_heads)
                        for layer in layers
                    ],
                    dim=1
                ),
                (3 * hidden_size, )
            )
        else:
            #weight
            query = layers[0].view(num_heads, hidden_size // num_heads, hidden_size)
            key = layers[1].view(num_key_value_heads, hidden_size // num_heads, hidden_size)
            value = layers[2].view(num_key_value_heads, hidden_size // num_heads, hidden_size)

            num_query_per_group = num_heads // num_key_value_heads
            qkv = [torch.cat([query[i*num_query_per_group:(i+1)*num_query_per_group], 
                key[i:i+1], value[i:i+1]], dim=0) for i in range(num_query_per_group)]

            #return torch.cat(qkv, dim=0).to(torch.bfloat16)
            qkv = torch.cat(qkv,dim=0).to(torch.bfloat16)
            return torch.reshape(qkv, (qkv.shape[0]*qkv.shape[1], -1))
            #return torch.reshape(torch.cat(qkv, dim=0), (qkv.shape[0] * qkv.shape[1], -1))
            # return torch.reshape(
            #     torch.cat(
            #         [
            #             layer.view(num_heads, 1, hidden_size // num_heads, hidden_size)
            #             for layer in layers
            #         ],
            #         dim=1
            #     ),
            #     (3 * hidden_size, hidden_size)
            # )

def find_transformers_weights_and_save_meg_ds_weights(
    meg_ds_filename: str,
    meg_ds_weight_names: List[str],
    opt_checkpoint_path: str,
    megatron_dump_folder_path:str,
    total_num_layers: int,
    num_heads: int,
    num_key_value_heads: int,
    hidden_size: int,
    trfs_weight_map: Dict[str, str]
):
    layer_id = get_layer_id(meg_ds_filename, total_num_layers=total_num_layers)
    trfs_weight_namess = {meg_ds_weight_name: get_transformers_weight_names(meg_ds_weight_name, layer_id=layer_id) for meg_ds_weight_name in meg_ds_weight_names}

    # Find the path they live in.
    trfs_filenames = set(trfs_weight_map[trfs_weight_name] for trfs_weight_names in trfs_weight_namess.values() for trfs_weight_name in trfs_weight_names)
    trfs_filename_to_weights = {
        trfs_filename: torch.load(os.path.join(opt_checkpoint_path, trfs_filename), map_location="cpu")
        for trfs_filename in trfs_filenames
    }

    # query those weights
    result = {
        meg_ds_weight_name: [
            trfs_filename_to_weights[trfs_weight_map[tfrs_weight_name]][tfrs_weight_name]
            for tfrs_weight_name in tfrs_weight_names
        ]
        for meg_ds_weight_name, tfrs_weight_names in trfs_weight_namess.items()
    }

    # possibly concatenate
    save_path = os.path.join(megatron_dump_folder_path, meg_ds_filename)
    with open(save_path, "wb") as fo:
        # qkv are mixed s.t. [q1 k1 v1 q2 k2 v2 ...] with (1,2..) being head_id
        torch.save(
            {
                key: merge_layers(values, num_heads=num_heads, 
                num_key_value_heads=num_key_value_heads, hidden_size=hidden_size)
                for key, values in result.items()
            },
            fo
        )


def convert_opt_checkpoint_to_megatron(
    opt_checkpoint_path: str,
    megatron_dump_folder_path: str,
    opt_index_path: str,
    num_proc: int
):
    # Get total number of layers
    with open(opt_index_path, "r") as fi:
        index_file = json.load(fi)["weight_map"]
    # Compute total amount of layers
    with open(os.path.join(opt_checkpoint_path, "config.json"), "r") as fi:
        config = json.load(fi)
    total_amount_of_layers = config["num_hidden_layers"]
    num_heads = config["num_attention_heads"]
    hidden_size = config["hidden_size"]
    num_key_value_heads = config["num_key_value_heads"]

    # Given the total number of layers we can compute exactly each meg_ds params we need to find.
    meg_ds_filename_to_meg_ds_weights = compute_meg_ds_weight_names(total_amount_of_layers)

    # Given the needed weights we can query them from the transformers checkpoint
    # We have to be smart about it and load a bin file once and get everything.
    if num_proc == 1:
        for meg_ds_filename, meg_ds_weight_names in tqdm(meg_ds_filename_to_meg_ds_weights.items()):
            find_transformers_weights_and_save_meg_ds_weights(
                meg_ds_filename=meg_ds_filename,
                meg_ds_weight_names=meg_ds_weight_names,
                opt_checkpoint_path=opt_checkpoint_path,
                megatron_dump_folder_path=megatron_dump_folder_path,
                total_num_layers=total_amount_of_layers,
                trfs_weight_map=index_file,
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                hidden_size=hidden_size
            )
    else:
        with Pool(num_proc) as pool:
            pool.starmap(
                partial(
                    find_transformers_weights_and_save_meg_ds_weights,
                    opt_checkpoint_path=opt_checkpoint_path,
                    megatron_dump_folder_path=megatron_dump_folder_path,
                    total_num_layers=total_amount_of_layers,
                    trfs_weight_map=index_file,
                    num_heads=num_heads,
                    num_key_value_heads=num_key_value_heads,
                    hidden_size=hidden_size
                ),
                tqdm(meg_ds_filename_to_meg_ds_weights.items())
            )

    # Create dummy mp_rank_00_model_states.pt
    torch.save(
        {
            "mp_world_size": 1,
            "module": None,
            "dp_world_size": 1,
            "checkpoint_version": 3,
            "iteration": 0
        },
        os.path.join(megatron_dump_folder_path, "mp_rank_00_model_states.pt")
    )

def main():
    args = get_args()
    convert_opt_checkpoint_to_megatron(
        opt_checkpoint_path=args.opt_checkpoint_path,
        megatron_dump_folder_path=args.megatron_dump_folder_path,
        opt_index_path=args.opt_sharded_index_path,
        num_proc=args.num_proc
    )

if __name__ == "__main__":
    main()
