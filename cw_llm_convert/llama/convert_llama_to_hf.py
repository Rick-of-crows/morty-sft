# coding=utf-8
import argparse
from convert_llama_original_checkpoint_to_pytorch_multiquery import convert_bloom_checkpoint_to_pytorch

parser = argparse.ArgumentParser()
# Required parameters
parser.add_argument(
    "--bloom_checkpoint_path",
    default=None,
    type=str,
    required=True,
    help="Path to the Megatron-LM checkpoint path.",
)
parser.add_argument(
    "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
)
parser.add_argument(
    "--bloom_config_file",
    default="",
    type=str,
    help=(
        "An optional config json file corresponding to the pre-trained model. \n"
        "This specifies the model architecture."
    ),
)
parser.add_argument(
    "--shard_model",
    action="store_true",
    help="An optional setting to shard the output model \nThis enables sharding the converted checkpoint",
)
parser.add_argument(
    "--pretraining_tp",
    default=4,
    type=int,
    help="Pretraining TP rank that has been used when training the model in Megatron-LM \n",
)

parser.add_argument(
    "--hidden_size",
    default=4,
    type=int,
    required=True,
    help="hidden state size\n",
)
parser.add_argument(
    "--num_attention_heads",
    default=4,
    type=int,
    required=True,
    help="number of attention heads \n",
)
parser.add_argument(
    "--num_key_value_heads",
    default=None,
    type=int,
    required=False,
    help="number of kv attention heads, use for multiquery \n",
)
args = parser.parse_args()

convert_bloom_checkpoint_to_pytorch(
    args.bloom_checkpoint_path,
    args.bloom_config_file,
    args.pytorch_dump_folder_path,
    args.shard_model,
    args.pretraining_tp,
    args,
)
