import json,re
import openai
import logging
import os
import numpy as np
import requests
import argparse
import tqdm
from multiprocessing import Pool, cpu_count

openai.api_key = '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'
openai.api_base = 'https://shanghai-chat-service-open-api.ai2open.win/v1'
openai.organization = 'org-ykL6XJ3eydRkR7ghoE92aJAK'
#directory = r'./chatgpt_evaluation_6tools_15'
#out_directory = r'./evaluate_logs'
judge_template = "得分: {score}分。{comment}"
def get_prompt(data):
    prompt = f"""You are a strict judge who are responsible for checking the quality of the answers and quality of its use of tools of an AI assistant named ADA. 
[History]

{data['history'] if data['history'] != "" else "empty"}

[Question] 

<Human> {data['instruction'].replace("<ADA>","<Final Response>").replace("</ADA>","</Final Response>")} </Human>

[The Start of ADA's Answer]

{data['output'].replace("<ADA>","<Final Response>").replace("</ADA>","</Final Response>")}

[The End of ADA's Answer]

[The Start of Reference Answer]

{data['reference'].replace("<ADA>","<Final Response>").replace("</ADA>","</Final Response>")}

[The End of Reference Answer] 

The response to the human is formated by "<Final Response>" and "</Final Response>".
We would like to request your feedback on the performance of ADA by comparing the ADA's Answer with the Reference Answer. 
The evaluation criteria are shown as follows:
1. Correctness of answer: 5 points. Directly compare the final response of the assistant and the reference answer, regardless of the quality and politeness of the answer. If the results of ADA is correct, 5 points should be given, otherwise 0 points.
2. Correctness of format: 3 points. Calling tool should follow the format like this "<Action> tool name </Action>\n<Action Input> a string based on JSON </Action Input>". Incorrect format, tool name, and tool input MUST result in 0 points. If the assistant does not call the tool, 3 points should be given. Checking this criterion regardless of the reference answer.
3. Appropriate use of tools: 2 points. This criterion is used to measure whether the tool is called at the right time. Note that as long as ADA attempts to call the tool unnecessarily but the reference answer does not, 0 points MUST be given. Please ignore other criteria when checking.

Please output a single line contaning a score and a comprehensive explanation of your evaluation for each criteria. The score and evaluation are separated by a space.
The order of output lines should match the order of the criteria metioned above.
For example:
```
0 Correctness of answer: The final response of ADA is empty, while the reference answer gives the correct result "23123乘以29382等于679399986。
0 Correctness of format: Calling tool format of ADA is correct, using <Action> <Action Input> tags to call tool and provide input. However, the tool name "The tool name to use MUST be one of [Calculator]." is invalid. 
0 Appropriate use of tools: Tool is called multiple times redundantly when only need to call once.
```
"""
    return prompt

def call_chatgpt(prompt):
    response = openai.ChatCompletion.create(
        #model="gpt-3.5-turbo",
        model="gpt-4-0314",
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message['content']

def process(item):
    prompt = get_prompt(item)
    #print(f"prompt = \n{prompt}")
    retry_count = 0
    while retry_count < 3:
        try:
            response = call_chatgpt(prompt)
            #print(f"response = \n{response}")
            res = re.findall("([0-9]) (.*?)\n", response+"\n", flags=re.DOTALL)
            if len(res) != 3:
                retry_count += 1
                continue
            return sum([int(s) for s,_ in res]), response
        except:
            retry_count += 1
    return 0,""

def main(args):
    with open(args.i, "r") as f:
        data = json.load(f)
    with Pool(5) as p:
        res = list(tqdm.tqdm(p.imap(process, data), total=len(data)))
        for i,(score,comment) in enumerate(res):
            data[i]['judgement1'] = judge_template.format(score=score,comment=comment)
        json.dump(data, open(args.o,"w"), ensure_ascii=False, indent=2)
    
    scores = []
    full_score = 0
    for di in data:
        score = float(re.findall("得分: (.*?)分", di["judgement1"])[0])
        scores.append(score)
        full_score += 10
    print(f"总分={sum(scores)}, 满分={full_score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("evaluate tool using by gpt")
    parser.add_argument("-i", help="lm eval", required=True)
    parser.add_argument("-o", help="result path", required=True)
    args = parser.parse_args()
    main(args)