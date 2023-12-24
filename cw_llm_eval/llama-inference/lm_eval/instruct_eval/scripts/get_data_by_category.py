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

def dotask(args):
    categories = args.category
    input_data = json_load(args.i)
    read_datas = []
    for item in input_data:
        i_type = item["type"]
        if i_type in categories:
            read_datas.append(item)
    json_dump(args.o, read_datas, ensure_ascii=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser("get data by category")
    parser.add_argument("-i", help="instruct path", required=True)
    parser.add_argument("-category", help="category list", nargs="+", required=True)
    parser.add_argument("-o", help="output json", required=True)
    args = parser.parse_args()
    dotask(args)