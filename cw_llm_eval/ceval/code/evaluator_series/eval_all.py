import os
import argparse
import pandas as pd
import torch
from evaluators.chatgpt import ChatGPT_Evaluator
from evaluators.moss import Moss_Evaluator
from evaluators.chatglm import ChatGLM_Evaluator
from evaluators.minimax import MiniMax_Evaluator
from evaluators.chatcw import ChatCW_Evaluator
import time
import json
choices = ["A", "B", "C", "D"]

def main(args):

    if "turbo" in args.model_name or "gpt-4" in args.model_name:
        evaluator=ChatGPT_Evaluator(
            choices=choices,
            k=args.ntrain,
            api_key=args.openai_key,
            model_name=args.model_name
        )
    elif "moss" in args.model_name:
        evaluator=Moss_Evaluator(
            choices=choices,
            k=args.ntrain,
            model_name=args.model_name
        )
    elif "chatcw" in args.model_name:
        evaluator=ChatCW_Evaluator(
            choices=choices,
            k=args.ntrain,
            model_name=args.model_name
        )
    elif "chatglm" in args.model_name:
        if args.cuda_device:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
        device = torch.device("cuda")
        evaluator=ChatGLM_Evaluator(
            choices=choices,
            k=args.ntrain,
            model_name=args.model_name,
            device=device
        )
    elif "minimax" in args.model_name:
        evaluator=MiniMax_Evaluator(
            choices=choices,
            k=args.ntrain,
            group_id=args.minimax_group_id,
            api_key=args.minimax_key,
            model_name=args.model_name
        )
    else:
        print("Unknown model name")
        return -1

    subject_name=args.subject
    if not os.path.exists(r"logs"):
        os.mkdir(r"logs")
    run_date=time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
    save_result_dir=os.path.join(r"logs",f"{args.model_name}_{run_date}")
    os.mkdir(save_result_dir)

    print(subject_name)

    val_sets=dic=json.load(open('subject_mapping.json', 'r', encoding='utf-8'))


    hard_set = {
        "advanced_mathematics",
        "discrete_mathematics", 
        "probability_and_statistics", 
        "college_chemistry", 
        "college_physics", 
        "high_school_mathematics",
        "high_school_chemistry",
        "high_school_physics"
    }
    hard_total = 0
    hard_correct = 0

    val_dic = {"STEM":[], "Social Science":[], "Humanities":[], "Other":[], "All":[]}

    for k,v in val_sets.items():
        val_dic[v[-1]].append(k)
        val_dic["All"].append(k)
     
    if subject_name not in val_dic.keys():
        print("Please input subject with STEM, Social Science, Humanities, Other")
        return -1

    total = 0
    correct = 0
    
    for sub_subject_name in val_dic[subject_name]:
        print("processing %s ..."%sub_subject_name)
        val_file_path=os.path.join('data/val',f'{sub_subject_name}_val.csv')
        val_df=pd.read_csv(val_file_path)
        if args.few_shot:
            dev_file_path=os.path.join('data/dev',f'{sub_subject_name}_dev.csv')
            dev_df=pd.read_csv(dev_file_path)
            correct_ratio = evaluator.eval_subject(sub_subject_name, val_df, dev_df, few_shot=args.few_shot,save_result_dir=save_result_dir,cot=args.cot)
        else:
            correct_ratio = evaluator.eval_subject(sub_subject_name, val_df, few_shot=args.few_shot,save_result_dir=save_result_dir)
        print("%s Acc:"%sub_subject_name,correct_ratio)
        total += correct_ratio[0]
        correct += correct_ratio[1]

        if sub_subject_name in hard_set:
            hard_total += correct_ratio[0]
            hard_correct += correct_ratio[1]
            
    print("Total Acc ", [total, correct, 100.0*correct/total])
    print("Hard Acc ", [hard_total, hard_correct, 100.0*hard_correct/hard_total])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--openai_key", type=str,default="xxx")
    parser.add_argument("--minimax_group_id", type=str,default="xxx")
    parser.add_argument("--minimax_key", type=str,default="xxx")
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--model_name",type=str)
    parser.add_argument("--cot",action="store_true")
    parser.add_argument("--subject","-s",type=str,default="operating_system")
    parser.add_argument("--cuda_device", type=str)    
    args = parser.parse_args()
    main(args)
