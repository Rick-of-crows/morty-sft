# -*- coding: utf-8 -*-
# @Time    : 2023/4/27 14:10
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : eval.py
# @Software: PyCharm
import os, sys
from tqdm import tqdm
import jsonlines
import argparse
import json
from multiprocessing import Pool

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from codes_eval.data_utils import read_problems, write_jsonl
from instruct_eval.instruct_models import get_model
from codes_eval.evaluation import evaluate_functional_correctness
from instruct_eval.utils import yaml_load


def gen_code_prompt(code_txt):
    prompt = "请根据函数注释补全下面的python函数, 使用markdown的形式展示补全后的代码：" + code_txt
    return prompt


def parse_model_out(res):
    res_sp = res.split("\n")
    do = False
    new_res = []
    for r in res_sp:
        if r.startswith("```python"):
            do = True
            continue
        if do:
            if r.startswith("```"):
                break
            new_res.append(r)
    new_res_str = "\n".join(new_res)
    return new_res_str


def get_gen_code(problem_path_or_sources, save_path, type_name, config, bs=1, i=0):
    # assert save_path.endswith(".jsonl"), "{} needs to end in .jsonl"
    if isinstance(problem_path_or_sources, list):
        problems = {}
        for line in problem_path_or_sources:
            if any(not x.isspace() for x in line):
                line_dict = json.loads(line)
                problems[line_dict["task_id"]] = line_dict
    else:
        problems = read_problems(problem_path_or_sources)
    problem_prompts = {}
    for task_id in problems:
        prom = problems[task_id]["prompt"]
        prom = gen_code_prompt(prom)
        problem_prompts[task_id] = prom
    # type_name = "chatglm"  ## type:模型类别，支持bloom, chatglm, openai, nemo, vicuna
    # config = {
    #     "model_path": "/nlp_data/zoo/pretrain/glm_model/6B/",
    #     "max_length": "384",
    #     "do_sample": "False"}
    if type_name == "chatglm" or type_name == "bloom":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
    ### do model init
    model = get_model(type_name, config)
    ### generate
    batch_text = []
    batch_task_id = []
    results = []

    total_num = len(problem_prompts)
    for k, task_id in enumerate(tqdm(problem_prompts, desc="pid:{}".format(i))):
        batch_task_id.append(task_id)
        batch_text.append(problem_prompts[task_id])
        if ((k + 1)) % bs == 0 or k == total_num - 1:
            ### pad for inference
            
            outs = model.base_generate([{"instruction":text, "history":""} for text in batch_text])
            if outs is None:
                continue
            assert len(outs) == len(batch_text)
            for i in range(len(batch_text)):
                r = {}
                r["task_id"] = batch_task_id[i]
                o = outs[i]
                o = parse_model_out(o)
                r["completion"] = o
                results.append(r)
            batch_task_id = []
            batch_text = []
    with jsonlines.open(save_path, "w") as fw:
        for res in results:
            fw.write(res)


def get_split_set(total_num, thread_num):
    avg_num = total_num // thread_num
    split_set_list = []

    for i in range(thread_num):
        start = i * avg_num
        end = (i + 1) * avg_num

        if i == thread_num - 1:
            end = total_num
        split_set_list.append((start, end))

    return split_set_list


def get_gen_code_multi_process(problem_path, save_path, type_name, config, pool_size, bs=1):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pool = Pool(pool_size)
    sources = []
    with open(problem_path, "r") as fr:
        for line in fr.readlines():
            sources.append(line)
    split_set_list = get_split_set(len(sources), pool_size)
    result = []
    for i in range(pool_size):
        save_path_i = save_path + str(i)
        start, end = split_set_list[i]
        print("pid:{} total num:{} start:{} end:{}".format(i, len(sources), start, end))
        result.append(pool.apply_async(get_gen_code,
                                       args=(sources[start:end], save_path_i, type_name, config, bs, i)))
        # get_gen_code(sources[start:end], save_path_i, type_name, config, bs, i)
    pool.close()
    pool.join()
    with open(save_path, "w") as fw:
        for i in range(pool_size):
            save_path_i = save_path + str(i)
            with open(save_path_i, "r") as fr:
                for line in fr.readlines():
                    fw.write(line)
            os.remove(save_path_i)


if __name__ == '__main__':
    """cmd
    python codes_eval/eval.py -data codes_eval/data/example_tiny.jsonl -config instruct_eval/configs/config.yml -player player1 -output result -bs 1
    """
    parser = argparse.ArgumentParser("instruct eval")
    parser.add_argument("-data", help="instruct data path", required=True, default="codes_eval/data/example_tiny_ch.jsonl")
    parser.add_argument("-config", help="model config path", required=True, default="instruct_eval/configs/config.yml")
    parser.add_argument("-player", help="which player do task", required=True, default="player1")
    parser.add_argument("-output", help="result path", required=True, default="result")
    parser.add_argument("-bs", help="batch size", type=int, default=1, required=False)
    parser.add_argument("-pool_size", help="pool size", type=int, default=2, required=False)
    args = parser.parse_args()
    problem_path = args.data
    model_config = yaml_load(args.config)
    assert args.player in model_config['player']
    val = model_config['player'][args.player]
    assert 'type' in val and 'config' in val
    type_name = val['type']
    config = val['config']
    bs = args.bs
    if type_name == "chatglm" or type_name == "bloom" or type_name == "vicuna":
        assert bs == 1, "chatglm/bloom/vicuna only support `bs` is 1"
    save_path = os.path.join(args.output, "code_gen_" + type_name + ".jsonl")
    print("problem path:{}".format(problem_path))
    print("save_path path:{}".format(save_path))
    # if not os.path.exists(save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if args.pool_size <= 1:
        get_gen_code(problem_path, save_path, type_name, config, bs=bs)
    else:
        get_gen_code_multi_process(problem_path, save_path, type_name, config, args.pool_size, bs=bs)
    results = evaluate_functional_correctness(save_path, problem_file=problem_path)
    print(results)
