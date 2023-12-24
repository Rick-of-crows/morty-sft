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
import requests

parser = ArgumentParser()
parser.add_argument("--ip", required=True, type=str, help="api ip address")
parser.add_argument("--port", required=True, type=str, help="api port")
parser.add_argument("--data_path", required=True, type=str, help="input data path")
parser.add_argument("--res_path", required=True, type=str, help="output path")

args = parser.parse_args()


def generate(inputs):
    """returns a list of zipped inputs, outputs and number of new tokens
    """
    headers = {"Content-Type": "application/json","User-Agent": "Apifox/1.0.0 (https://www.apifox.cn)","Accept": "*/*","Connection":"keep-alive"}
    api_addr = 'http://{}:{}/generate'.format(args.ip, args.port)


    
    #responses = requests.request("POST", api_addr, data=json.dumps(inputs), headers=headers)
    while True:
        try:
            responses = requests.post(api_addr, data=json.dumps(inputs), headers=headers,timeout=30)
            break
        except:
            print("Connection Error")
            time.sleep(3)
            continue
        
    #print(responses)
    response = ""
    new_tokens = ""
    if responses.status_code == 200:
        response = responses.json()["text"][0].strip("\n")
        new_tokens = str(responses.json()["num_generated_tokens"][0])
    else:
        print("error code: ", responses.status_code)

    return (inputs["text"], response, new_tokens)


def main():
    # hardcode here by yanc
    test_data_list = [f"{args.data_path}/honest/honest.json", f"{args.data_path}/harmness/harmness.json",f"{args.data_path}/helpfulness/helpfulness.json"]
    # 2, load data
    for test_data_path in test_data_list:
        print(f"*** Loading the data {test_data_path}")
        try:
            data = json.load(open(test_data_path, "r", encoding="utf-8"))
        except:
            data=[]
            with open(test_data_path, 'r', encoding='utf-8') as fr:
                for line in fr:
                    data.append(json.loads(line.strip('\n')))
    
        # 3, generate
        num_tokens = 512
        res = []
        total_time = []

        # if args.batch_size > len(input_sentences):
        #     # dynamically extend to support larger bs by repetition
        #     input_sentences *= math.ceil(args.batch_size / len(input_sentences))

        # , repetition_penalty=1.2
        template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
        for d in data:
            instruction = d["instruction"]
            input = d.get("input", "")
            if input:
                model_input = f"{template}### Instruction:\n{instruction}\n\n{input}\n### Response:\n"
            else:
                model_input = f"{template}### Instruction:\n{instruction}\n### Response:"
            
            # XXX: this is currently doing world_size streams on world_size gpus, so we can feed it different inputs on each! and hence the time can be divided by world_size
            t_generate_start = time.time()
            inputs = {"tokens_to_generate": num_tokens, "do_sample":False, "bos_token_id": 1, "eos_token_id":2, "pad_token_id":3, "repetition_penalty":1.1, "text":[model_input] }
            pairs = generate(inputs)

            t_generate_span = time.time() - t_generate_start
            total_time.append(t_generate_span)
            print(f"{'-'*60}\n{pairs[1]}\n")
            print(t_generate_span)
            d.update({"output": pairs[1]})
            print("updated...")
            
        # 4, write res
        result_output = args.res_path + '/result-' + test_data_path.split('/')[-1]
        with open(result_output, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            '''
            for d in data:
                json.dump(d, f, ensure_ascii=False)
                f.write('\n')
            '''


if __name__ == "__main__":
    main()
