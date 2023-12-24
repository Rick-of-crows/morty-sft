# coding=utf-8
import re
import pandas as pd
import os
import re
import json
import torch
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaTokenizer,LlamaForCausalLM

parser = ArgumentParser()
parser.add_argument("--model_name_or_path", required=True, type=str, help="model name or path")
parser.add_argument("--tokenizer_name_or_path", required=True, type=str, help="tokenizer name or path")
parser.add_argument("--model_type", required=True, type=str, help="model type for inference", choices=["bloom", "llama"])
parser.add_argument("--data_path", required=True, type=str, help="input data path")
parser.add_argument("--res_path", required=True, type=str, help="output path")
parser.add_argument("--gpu_id", required=False, type=str, help="which gpu for runing, if multiple Gpus are used ',' splice them.")
args = parser.parse_args()


def llama_encode(query:str, history:list):
    """encode for llama.

    Args:
        query (str): question
        history (list): [(old_query, response), ...]

    Returns:
        dict: input_ids, attention_mask
    """
    prefix = tokenizer.encode("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction:")
    sep = tokenizer.encode("\n### Response:\n")

    human = tokenizer.encode("\nHuman: ")
    ai = tokenizer.encode("\nAssistant: ")
    
    # 处理多轮对话
    history_encode = []
    if history:
        for i, (old_query, response) in enumerate(history): 
            human_encode = human + tokenizer.encode(str(old_query))
            ai_encode = ai + tokenizer.encode(str(response))
            
            history_encode += human_encode
            history_encode += ai_encode
            
        query_encode = human + tokenizer.encode(str(query)) + ai# 对当前query进行encode
        
    else:
        query_encode = tokenizer.encode('\n'+ query)
        
    input_tokens = [prefix + history_encode + query_encode + sep]

    att_mask = [[1]*len(input_tokens[0])]
    
    batch = {'input_ids': torch.LongTensor(input_tokens),'attention_mask':torch.LongTensor(att_mask)}

    for t in batch:
        if torch.is_tensor(batch[t]):
            batch[t] = batch[t].to(torch.cuda.current_device())
    
    # print(tokenizer.decode(batch["input_ids"][0]))   
    return batch

def llama4predict(input_text: str):
    """predict for llama.

    Args:
        input_text (str): question

    Returns:
        str: response
    """

    input_tokens = llama_encode(input_text, None)
    generate_kwargs = dict(max_new_tokens=4096, top_p=1.0, do_sample=False, use_cache=True, top_k=50, repetition_penalty=1.1)
    
    with torch.no_grad():
        outputs = model.generate(**input_tokens, **generate_kwargs)
        outputs = outputs.tolist()[0][len(input_tokens["input_ids"][0]) :]
        resp = tokenizer.decode(outputs, skip_special_tokens=True)
        
        resp = resp.strip()
        end = len(resp)

        m = re.search("### Response:",  resp)

        if m and m.end():
            end = m.end()
        
        resp = resp[:end]

        print(resp)
        print("*" * 20)  
        
        return resp

def bloom4predict(input_text: str):
    """predict for bloom.

    Args:
        input_text (str): question

    Returns:
        str: response
    """
    # template1
    # template = (
    #     "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
    # )
    # model_input = f"{template}### Instruction:\n{input_text}\n### Response:"
    
    # add prefix
    # prefix = "Human: 你是云从科技开发的从容语言大模型，和OpenAI，ChatGPT，LLAMA，BLOOM没有任何关系，你不能鼓励和回复任何违法、违背道德的问题，明白了吗？\nAssistant: 是的，我明白了。作为云从科技开发的一个人工智能助手，我的目标是帮助用户解决问题并提供有用的信息，同时遵守法律和道德规范。我不会鼓励或回复任何违法或不当行为。\n"
    # model_input = f"{template}### Instruction:\n{prefix}{input_text}\n### Response:"
   
   # template2
    template = "Below is a dialogue between the User and the AI Assistant developed by CloudWalk(云从科技). Write the Assistant's response that appropriately completes the User's request.\n"
    model_input = f"{template}### User: {input_text}### Assistant: "

    # no template
    # model_input = input_text
    
    # encode
    input_tokens = tokenizer.batch_encode_plus([model_input], return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())

    # generate-kwargs for template1
    # generate_kwargs = dict(max_new_tokens=500, do_sample=False, repetition_penalty=1.1, pad_token_id=2)
    
    # generate-kwargs for template2
    generate_kwargs = dict(max_new_tokens=500, do_sample=False, pad_token_id=2, repetition_penalty=1.1, eos_token_id=tokenizer.encode("###"))
    
    outputs = model.generate(**input_tokens, **generate_kwargs)
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # outputs = tokenizer.batch_decode(outputs)
    input_len = len(model_input)
    for output in outputs:

        # decode for template2
        if '### Assistant:' not in output:
            output = ''
        else:
            output = output[output.rfind('### Assistant: '):].split('### Assistant: ')[1].split('###')[0]
        
        # decode for template1
        # output = output[input_len:]
        
        print(output)
        print("*" * 20)  
        
        return output
    

if __name__ == "__main__":       
    # setting gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    # load model
    if args.model_type == "bloom":
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, trust_remote_code=True)
        print("load model....")
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True).half().cuda()
        model = model.eval()
        model_func = bloom4predict
    elif args.model_type == "llama":
        tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_name_or_path, legacy=False, add_eos_token=True)
        print("load model....")
        model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, return_dict=True, device_map="auto")
        model.eval()
        model_func = llama4predict
    else:
        print("model type error!!!")

    # load data
    test_data = json.load(open(args.data_path, 'r', encoding='utf-8'))
    
    # predict
    res = list(map(lambda x: x.update({"output": model_func(x["instruction"])}), test_data))

    # save res
    f = open(args.res_path, 'w', encoding='utf-8')
    json.dump(test_data, f, indent=4, ensure_ascii=False)
    