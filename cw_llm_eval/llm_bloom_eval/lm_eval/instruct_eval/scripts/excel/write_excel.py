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


def write_excel(args):
    inp = args.i
    data_items = json_load(inp)
    save_dict = {'instruction':[]}
    for item in data_items:
        ins = item["instruction"]
        save_dict['instruction'].append(ins)
        #fd_w.write(ins)

    df = pd.DataFrame(save_dict)
    df.to_excel(args.o)

if __name__=='__main__':
    parser = argparse.ArgumentParser("get instruction from json and save to excel")
    parser.add_argument("-i", help="input json file", required=True)
    parser.add_argument("-o", help="output excel file", required=True)
    args = parser.parse_args()
    write_excel(args)