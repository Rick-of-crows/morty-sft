import pandas as pd
import json
import os, sys, argparse

def json_load(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def json_dump(file, data, ensure_ascii=True):
    with open(file, "w", encoding="utf-8") as f:
        if ensure_ascii:
            json.dump(data, f, indent=2)
        else:
            json.dump(data, f,ensure_ascii=False,indent=2)

def process_web_data(args):
    input_file = args.i
    output_file = args.o
    read_datas = []
    count = 0
    with open(input_file, 'r', encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            ## skip多轮
            if len(data["history"]) > 1:
                continue
            history = data["history"][0]
            #print("prompt:", history['prompt'])
            #print("history['prompt']", history)
            data_dict = {
                "instruction": history['prompt'],
                "lang": "CN",
                "output": "",
                "question_num": count,
                "source": "web",
                "type": "Universal",
                #"resp1": history['ai'],
            }
            read_datas.append(data_dict)
            count += 1
    print(f"total process {count} infos")
    json_dump(output_file, read_datas, ensure_ascii=False)

if __name__=='__main__':
    parser = argparse.ArgumentParser("convert txt web file to data json file")
    parser.add_argument("-i", help="input file", required=True)
    parser.add_argument("-o", help="output json file", required=True)
    args = parser.parse_args()
    process_web_data(args)