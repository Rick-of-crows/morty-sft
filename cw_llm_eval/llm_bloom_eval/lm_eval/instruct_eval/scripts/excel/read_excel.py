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

def process_excel(args):
    excel_file = args.i
    question_file = args.q
    output = args.o
    ## get excel dict
    df = pd.read_excel(excel_file)
    excel_data = df.values

    excel_d = {}
    for ex_d in excel_data:
        #print("ex_d:", ex_d)
        #instruct, answer = ex_d
        #_, instruct, _, answer, _ = ex_d
        _, instruct, answer = ex_d
        #print("instruct:", instruct, " answer:", answer)
        excel_d[instruct.strip()] = str(answer).strip()

    ## match question
    t_data = json_load(question_file)
    count = 0
    for t in t_data:
        ins = t["instruction"].strip()
        if ins not in excel_d:
            print("count:", count, " not found:", ins)
            t["output"] = ""
            continue
        t["output"] = excel_d[ins]
        count += 1
    ## save file
    json_dump(output, t_data, ensure_ascii=False)

if __name__=='__main__':
    parser = argparse.ArgumentParser("Get model answer from excel and save to json file")
    parser.add_argument("-i", help="input excel file", required=True)
    parser.add_argument("-q", help="json file to match question", required=True)
    parser.add_argument("-o", help="output json file", required=True)
    args = parser.parse_args()
    process_excel(args)