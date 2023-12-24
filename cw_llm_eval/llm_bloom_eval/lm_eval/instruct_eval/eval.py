#-- codeing:utf-8
import instruct_models
import os, sys, argparse
import codecs
import importlib
importlib.reload(sys)
from progressbar import *
import multiprocessing

from tqdm import tqdm
from utils import *

import metrics

kQSize = 1024

def score_worker(val, q_in, q_out):
    type_name = val['type']
    config = val['config']
    model = instruct_models.get_model(type_name, config)
    model.init_score_by_category()
    while True:
        item = q_in.get()
        if item is None:
            break
        data_in, idx, results = item
        data = data_in[idx]
        category = data["type"]
        score_result = []
        for k in range(len(results)):
            player_info = results[k]
            #print("player:", player_info["instruction"], " data:", data["instruction"])
            assert player_info["instruction"] == data["instruction"], " question_num:" + str(data["question_num"])
            instruction = add_instruct_limit(player_info, player=False)
            player_answer = player_info["output"]
            data["resp"+str(k+1)] = player_answer
            ### debug category
            # if category not in ["Generation"]:
            #     data["judgement"+str(k+1)] = ""
            #     score_result.append(None)
            #     continue
            judgement = model.score_generate(instruction, player_answer, data["output"], category)
            #print("judgement:",judgement)
            score_result.append(judgement[0])
            data["judgement"+str(k+1)] = judgement[1]
        q_out.put((data, idx, score_result))

def rank_worker(val, q_in, q_out):
    type_name = val['type']
    config = val['config']
    rank_flip = val['rank_flip'] if 'rank_flip' in val else False
    model = instruct_models.get_model(type_name, config)
    if val['metric'] == "rank_old":
        model.init_chain_rank()
    elif val['metric'] == "rank":
        model.init_rank_by_category()
    while True:
        item = q_in.get()
        if item is None:
            break
        data_in, idx, results = item
        data = data_in[idx]
        category = data["type"]
        rank_result = []
        assert len(results) == 2
        player1_info = results[0]
        player2_info = results[1]
        assert player1_info["instruction"] == data["instruction"], " question_num:" + str(data["question_num"])
        assert player2_info["instruction"] == data["instruction"], " question_num:" + str(data["question_num"])
        instruction = add_instruct_limit(player1_info, player=False)
        player1_answer = player1_info["output"]
        player2_answer = player2_info["output"]
        data["resp1"] = player1_answer
        data["resp2"] = player2_answer
        ### val['metric'] in ["rank", "rank_old"]
        generate_func = getattr(model, val['metric'] + "_generate")
        judgement = generate_func(instruction, player1_answer, player2_answer, data["output"], category)
        data["rank"] = judgement[0]
        data["judgement"] = judgement[1]
        rank_result = judgement[0]
        if rank_flip:
            judgement_flip = generate_func(instruction, player2_answer, player1_answer, data["output"], category)
            data["rank_flip"] = judgement_flip[0]
            data["judgement_flip"] = judgement_flip[1]
            rank_result = merge_jugdement(judgement[0], judgement_flip[0])
        q_out.put((data, idx, rank_result))

def metric_worker(args, key, metric, data, q_out):
    save_path = os.path.join(args.output, key+".json")
    result_path = os.path.join(args.output, key+"_result.json")
    ## pbar
    widgets = ['Progress: ', Percentage(), ' ', Bar('#'),' ', Timer(),' ', ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=len(data))
    pbar.start()
    count = 0
    while True:
        item = q_out.get()
        if item is not None:
            data_out, idx, result = item
            data[idx] = data_out
            #print("result:", result, " data:", data_out)
            metric.process_result(result, data_out)
            pbar.update(count)
            count += 1
        else:
            break
    pbar.finish()
    ## save result
    json_dump(save_path, data, ensure_ascii=False)
    metric.print_and_save_result(result_path)

if __name__=='__main__':
    parser = argparse.ArgumentParser("instruct eval")
    parser.add_argument("-data", help="instruct data path", required=True)
    parser.add_argument("-config", help="model config path", required=True)
    parser.add_argument("-output", help="result path", required=True)
    args = parser.parse_args()
    data = json_load(args.data)
    results = []
    model_config = yaml_load(args.config)
    set_dir(args.output)
    ### player
    for key in model_config['player']:
        save_path = os.path.join(args.output, key+".json")
        val = model_config['player'][key]
        #type_name = val['type']
        #config = val['config']
        if "load_from_path" in val and os.path.isfile(val['load_from_path']):
            print(key, " load answer from path:", val['load_from_path'])
            result = json_load(val['load_from_path'])
            results.append(result)
            continue
        bs = val["batchsize"] if 'batchsize' in val else 1
        run_player_cmd(key, args, bs)
        result = json_load(save_path)
        results.append(result)
    ### reviewer
    for key in model_config['reviewer']:
        val = model_config['reviewer'][key]
        ### 根据instruct_data选取player回答的子集进行评估.
        results = select_results_by_data(results, data)
        # exit()
        metric = metrics.get_metric(val['metric'], len(results))
        worker = val['worker'] if 'worker' in val else 1
        assert worker > 0
        q_in = [multiprocessing.Queue(kQSize) for i in range(worker)]
        q_out = multiprocessing.Queue(kQSize)
        run_worker = score_worker if val['metric'] == "score" else rank_worker
        ps = [multiprocessing.Process(target=run_worker, args=(val, q_in[i], q_out,)) for i in range(worker)]
        for p in ps:
            p.start()
        metric_p = multiprocessing.Process(target=metric_worker, args=(args, key, metric, data, q_out,))
        metric_p.start()
        for idx in range(len(data)):
            rr = []
            for k in range(len(results)):
                rr.append(results[k][idx])
            q_in[idx % len(q_in)].put((data, idx, rr))
        for q in q_in:
            q.put(None)
        for p in ps:
            p.join()
        q_out.put(None)
        metric_p.join()
