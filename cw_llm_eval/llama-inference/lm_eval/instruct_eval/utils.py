
import json
import yaml
import os, sys

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

def yaml_load(file):
    with open(file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data

def merge_jugdement(y1, y2):
    if y1 == 0 and y2 == 1:
        return 0
    elif y1 == 1 and y2 == 0:
        return 1
    else:
        return -1

def prompt_dict_mapping(prompt_dict):
    prompt_map = {
        "OpenQA": ["Art", "Biology", "Chemical", "Chinese", "Geography", "History", "Math_qa", "Music", "Physics", "Sports"],
        "Toxicity": ["Attack", "Bias", "Control", "Hate", "Politics", "Violence"]
    }
    for key in prompt_map.keys():
        val = prompt_dict[key]
        map_lists = prompt_map[key]
        for new_key in map_lists:
            prompt_dict[new_key] = val
    return prompt_dict

def add_instruct_limit(item, player=False):
    instruct = item["instruction"]
    lang_type = item["lang"]
    category = item["type"]
    # if lang_type=="CN":
    #     ### 仅对player回答作限制，不作为reviewer评判标准.
    #     if player:
    #         instruct = "回答要求尽可能精简并且字数限制在200字以内。" + instruct
    #     if category != "Translation":
    #         instruct = "要求使用中文作答。" + instruct
    # elif lang_type=="EN":
    #     if player:
    #         instruct = "limit the answer to within 200 words." + instruct
    #     if category != "Translation":
    #         instruct = "Please answer in English." + instruct
    # else:
    #     raise RuntimeError('Unknown lang_type: %s'% lang_type)
    return instruct

def method_call(o, method_name):
    if not hasattr(o, method_name):
        raise RuntimeError('Unknown method_name:%s'% method_name)
    func = getattr(o, method_name)
    return func

def set_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

def run_player_cmd(player, args, bs):
    cmd_str = "python player.py -data {} -config {} -player {} -output {} -bs {}".format(args.data, args.config, player, args.output, bs)
    print("run player cmd:", cmd_str)
    os.system(cmd_str)