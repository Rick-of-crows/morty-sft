import math
import os
import time
from argparse import ArgumentParser
import numpy as np
import torch
import torch.distributed as dist
import json
import deepspeed
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


parser = ArgumentParser()
parser.add_argument("--model_name_or_path", required=True, type=str, help="model name or path")
parser.add_argument("--tokenizer_name_or_path", required=True, type=str, help="tokenizer name or path")
parser.add_argument("--data_path", required=True, type=str, help="input data path")
parser.add_argument("--res_path", required=True, type=str, help="output path")
parser.add_argument("--gpu_id", required=False, type=str, help="which gpu for runing")
parser.add_argument("--local_rank", required=False, type=int, help="used by dist launchers")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--cpu_offload", action="store_true", help="whether to activate CPU offload")
parser.add_argument("--nvme_offload_path", help="whether to activate NVME offload and the path on nvme")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

deepspeed.init_distributed("nccl")
rank = dist.get_rank()
local_rank = int(os.getenv("LOCAL_RANK", "0"))

def print_rank0(*msg):
    if rank != 0:
        return
    print(*msg)


def create_model():
    """
    load model and deepspeed engine
    """
    # 1, load model
    print_rank0(f"*** Loading the model {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    model = model.eval()

    # 2, init ds-engine
    # XXX: can't automatically derive dtype via config's `from_pretrained`
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    model_hidden_size = config.hidden_size
    train_batch_size = 1 * world_size
    ds_config = {
        "fp16": {
            "enabled": True,
        },
        "zero_optimization": {
            "stage": 0,
            # "overlap_comm": True,
            # "contiguous_gradients": True,
            # "reduce_bucket_size": model_hidden_size * model_hidden_size,
            # "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
            # "stage3_param_persistence_threshold": 0,
        },
        "steps_per_print": 2000,
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": 1,
        "wall_clock_breakdown": False,
    }

    if args.cpu_offload and args.nvme_offload_path:
        raise ValueError("Use one of --cpu_offload or --nvme_offload_path and not both")

    if args.cpu_offload:
        ds_config["zero_optimization"]["offload_param"] = dict(device="cpu", pin_memory=True)

    if args.nvme_offload_path:
        ds_config["zero_optimization"]["offload_param"] = dict(
            device="nvme",
            pin_memory=True,
            nvme_path=args.nvme_offload_path,
            buffer_size=4e9,
        )
    print_rank0(ds_config)

    # create ds-engine
    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    ds_engine.module.eval()
    model = ds_engine.module

    return model, tokenizer


def generate(inputs, model, tokenizer, generate_kwargs):
    """returns a list of zipped inputs, outputs and number of new tokens

    :param inputs (List[Text]) : input
    :param model
    :param tokenizer
    :param generate_kwargs
    """
    input_tokens = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())

    outputs = model.generate(**input_tokens, **generate_kwargs)

    input_tokens_lengths = [x.shape[0] for x in input_tokens.input_ids]
    output_tokens_lengths = [x.shape[0] for x in outputs]

    total_new_tokens = [o - i for i, o in zip(input_tokens_lengths, output_tokens_lengths)]
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return zip(inputs, outputs, total_new_tokens)


def main():
    # 1, create model
    model, tokenizer = create_model()

    # 2, load data
    print_rank0(f"*** Loading the data {args.data_path}")
    try:
        data = json.load(open(args.data_path, "r", encoding="utf-8"))
    except:
        data=[]
        with open(args.data_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                data.append(json.loads(line.strip('\n')))
    
    # 3, generate
    num_tokens = 500
    print(f"*** Starting to generate {num_tokens} tokens with bs={args.batch_size}")
    res = []
    total_time = []

    # if args.batch_size > len(input_sentences):
    #     # dynamically extend to support larger bs by repetition
    #     input_sentences *= math.ceil(args.batch_size / len(input_sentences))

    # , repetition_penalty=1.2
    generate_kwargs = dict(max_new_tokens=num_tokens, do_sample=False, pad_token_id=2, repetition_penalty=1.1)
    template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
    for d in data:
        instruction = d["instruction"]
        input = d.get("input", "")
        if input:
            model_input = f"{template}### Instruction:\n{instruction}\n\n{input}\n### Response:\n"
        else:
            model_input = f"{template}### Instruction:\n{instruction}\n### Response:\n"
            
        # XXX: this is currently doing world_size streams on world_size gpus, so we can feed it different inputs on each! and hence the time can be divided by world_size
        t_generate_start = time.time()
        pairs = generate([model_input], model, tokenizer, generate_kwargs)
        t_generate_span = time.time() - t_generate_start
        total_time.append(t_generate_span)
        for i, o, _ in pairs:
            o = o[len(i):]
            print(f"{'-'*60}\n{o}\n")
            print(t_generate_span)
            d.update({"output": o})
            
        # break
    print_rank0(f"average generate time: {np.mean(total_time)}")

    # 4, write res
    with open(args.res_path, "w", encoding="utf-8") as f:
        # json.dump(data, f, ensure_ascii=False, indent=4)
        for d in data:
            json.dump(d, f, ensure_ascii=False)
            f.write('\n')

    return data


if __name__ == "__main__":
    main()
