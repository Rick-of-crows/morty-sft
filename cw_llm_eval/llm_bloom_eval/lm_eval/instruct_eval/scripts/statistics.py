import json

import os, sys
import random

path = "/home/wangxi/nlp/lm_eval/lm_eval/instruct_eval/instruct_data/harmness/harmness.json"
#path = "mini_test.json"
#path = "/home/wangxi/nlp/lm_eval/lm_eval/instruct_eval/instruct_data/cw-eval.json"
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

dd = json_load(path)

print(len(dd))

res = {}
for item in dd:
    i_type = item["type"]
    #print("instruction:", item["instruction"])
    if i_type not in res:
        res[i_type] = 0
    res[i_type] += 1
for key in res:
    print(key,":", res[key])

### random for mini test
'''max_select = 9
save_list = []
random.shuffle(dd)
for key in res:
    count = 0
    for item in dd:
        #print(count, max_select)
        if count > max_select:
            break
        if item["type"] != key:
            continue
        save_list.append(item)
        count += 1
save_path = "mini_test.json"
json_dump(save_path, save_list, ensure_ascii=False)'''
