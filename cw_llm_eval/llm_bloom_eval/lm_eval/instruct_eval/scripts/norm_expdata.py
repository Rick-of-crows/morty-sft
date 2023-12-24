import os, sys, argparse
#import pandas as pd

def norm_dict(result_dict):
    assert "满分" in result_dict
    max_scores = [score for score in result_dict["满分"]]
    #print(max_scores)
    for key in result_dict:
        if key == "模型":
            continue
        vals = result_dict[key]
        for i, val in enumerate(vals):
            #print("i:",i, " val:", val, " max_scores:", max_scores[i])
            score = val / max_scores[i] * 100.0
            score = round(score, 1)
            #print(score)
            result_dict[key][i] = score
    #print("norm_result:", result_dict)
    ### debug
    for key in result_dict:
        vals = result_dict[key]
        if key == "模型":
            continue

        sum_score = round(sum(vals[1:]), 1)
        avg_score = sum_score / len(vals[1:])
        avg_score = round(avg_score, 1)
        
        #print("model:", key, " total_score:", vals[0], " sum_score:", sum_score, " avg_score:", avg_score)
        result_dict[key][0] = avg_score
    return result_dict

def process_markdown(lines, key_name, is_norm):
    start_line = -1
    end_line = -1
    result_dict = {}
    class_names = []
    for idx, line in enumerate(lines):
        infos = line.strip().split(" ")
        if len(infos) > 2 and infos[1] == key_name:
            start_line = idx
        ### table head
        if start_line > 0 and start_line + 1 == idx:
            model_infos = line.strip().split("|")
            assert len(model_infos) > 1
            result_dict[model_infos[0]] = []
            for col in range(1, len(model_infos)):
                if model_infos[col] == "":
                    continue
                result_dict[model_infos[0]].append(model_infos[col])
                class_names.append(model_infos[col])
            #print("result_dict:", result_dict)
        ### model score
        if start_line > 0 and idx > start_line + 2:
            model_scores = line.strip().split("|")
            model_key = model_scores[0].strip()
            if len(model_scores) == 1:
                end_line = idx
                break
            result_dict[model_key] = []
            for col in range(1, len(model_scores)):
                if model_scores[col] == "":
                    continue
                score = float(model_scores[col])
                result_dict[model_key].append(score)
            assert(len(result_dict[model_key]) == len(class_names))
            ### check score sum
            vals = result_dict[model_key]
            assert vals[0] == sum(vals[1:]), "key:%s, vals:%s, sum:%s"%(model_key, vals[0], sum(vals[1:]))
    #print("result_dict:", result_dict)
    if is_norm:
        result_dict = norm_dict(result_dict)
    return result_dict, start_line, end_line

def dict2markdown(score_d):
    process_lines = []
    model_list = score_d["模型"]
    model_list = ["模型"] + model_list
    #print(model_list)
    for key in score_d:
        if key == "模型":
            continue
        score_list = score_d[key]
        score_list = [key] + score_list
        process_lines.append(score_list)
        #print(score_list)
    sorted_lines = sorted(process_lines, key=lambda x:x[1], reverse=True)
    ret_lines = []
    ret_lines.append("|".join(model_list) + "\n")
    ret_lines.append("|".join(["---"]*len(model_list)) + "\n")
    for line_list in sorted_lines:
        for i, s in enumerate(line_list):
            if isinstance(s, float):
                line_list[i] = str(s)
        #print("xx:", line_list)
        string = "|".join(line_list) + "\n"
        ret_lines.append(string)
    #print(ret_lines)
    return ret_lines

def write_norm_file(lines, output):
    w_fd = open(output, "w")
    for line in lines:
        w_fd.write(line)
    w_fd.close()

def final_rank(lines, help_dict, honest_dict, harmness_dict, is_norm):
    help_len = len(help_dict["模型"]) - 1
    honest_len = len(honest_dict["模型"]) - 1
    harm_len = len(harmness_dict["模型"]) - 1
    total_len = help_len + honest_len + harm_len
    if is_norm:
        help_weight = round(help_len / total_len, 3)
        honest_weight = round(honest_len / total_len, 3)
        harm_weight = 1.0 - help_weight - honest_weight
        harm_weight = round(harm_weight, 3)
        total_weight = help_weight + honest_weight + harm_weight
        assert total_weight == 1.0
        weight_list = ["权重", str(total_weight), str(help_weight), str(honest_weight), str(harm_weight)]
    else:
        help_weight = honest_weight = harm_weight = 1.0
    #print(help_weight, honest_weight, harm_weight, weight_list)
    sorted_lines = []
    final_rank_list = []
    final_rank_head = ["模型", "总分", "帮助性<br>helpfulness", "真实性<br>honest", "无害性<br>harmness"]
    final_rank_seq = "|".join(["---"] * len(final_rank_head)) + "\n"
    final_rank_list.append("|".join(final_rank_head) + "\n")
    final_rank_list.append(final_rank_seq)
    if is_norm:
        final_rank_list.append("|".join(weight_list) + "\n")
    for key in help_dict:
        key = key.strip()
        if key == "模型":
            continue
        help_score = help_dict[key][0]
        honest_score = honest_dict[key][0]
        harm_score = harmness_dict[key][0]
        weight_score = float(help_score) * help_weight + float(honest_score) * honest_weight + float(harm_score) * harm_weight
        weight_score = round(weight_score, 1)
        #print("key:", key, " vals:", help_dict[key])
        model_score_list = [key, weight_score, help_score, honest_score, harm_score]
        sorted_lines.append(model_score_list)
        #print("key:", key, " help:", help_score, " honest:", honest_score, " harm:", harm_score, " weight:", weight_score)
    ## 
    sorted_lines = sorted(sorted_lines, key=lambda x:x[1], reverse=True)
    for line_list in sorted_lines:
        for i, s in enumerate(line_list):
            if isinstance(s, float):
                line_list[i] = str(s)
        string = "|".join(line_list) + "\n"
        final_rank_list.append(string)
    #print("final_rank_list:", final_rank_list)
    ### locate final rank line
    start_idx = -1
    for idx, line in enumerate(lines):
        infos = line.strip().split(" ")
        if len(infos) > 1 and infos[1] == "总榜单":
            start_idx = idx
            break
    return final_rank_list, idx

def replace_link(lines, is_norm):
    if not is_norm:
        return lines
    for idx, line in enumerate(lines):
        if "experiments_data.md" in line:
            line = line.replace("归一化后", "归一化前")
            line = line.replace("norm_experiments_data.md", "experiments_data.md")
            lines[idx] = line
            break
    return lines

def process(args):
    input_file = args.i
    fd = open(input_file, "r")
    lines = fd.readlines()
    ## replace link line
    lines = replace_link(lines, args.norm)
    ## get table dict and start/end line index
    help_dict, help_start, help_end = process_markdown(lines, "helpfulness", args.norm)
    honest_dict, honest_start, honest_end = process_markdown(lines, "honest", args.norm)
    harmness_dict, harm_start, harm_end = process_markdown(lines, "harmness", args.norm)
    ### sort dict and markdown format
    help_lines = dict2markdown(help_dict)
    honest_lines = dict2markdown(honest_dict)
    harmness_lines = dict2markdown(harmness_dict)
    ## replace old infos
    lines[help_start + 1:help_end] = help_lines
    lines[honest_start + 1:honest_end] = honest_lines
    lines[harm_start + 1:harm_end] = harmness_lines
    ## get final rank lines
    final_rank_lines, start_idx = final_rank(lines, help_dict, honest_dict, harmness_dict, args.norm)
    lines[start_idx + 1:start_idx + 1 + len(final_rank_lines)] = final_rank_lines
    # write final result
    write_norm_file(lines, args.o)
    fd.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser("norm score from exp data")
    parser.add_argument("-i", help="input file", required=True)
    parser.add_argument("-norm", action="store_true", help="norm score")
    parser.add_argument("-o", help="output file", required=True)
    args = parser.parse_args()
    process(args)
