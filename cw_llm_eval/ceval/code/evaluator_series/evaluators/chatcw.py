import os
from tqdm import tqdm
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
from evaluators.evaluator import Evaluator
from time import sleep
import re

def get_ans(gen_ans):
    answer_patterns = [
            r'([ABCD])是正确答案',
            r'选项([ABCD])正确',
            r'正确选项是([ABCD])',
            r'([ABCD])选项正确',
            r'答案是([ABCD])',
            r'答案为([ABCD])',
            r'答案([ABCD])',
            r'选择([ABCD])',
            r'答案：([ABCD])',
            r'选择答案([ABCD])',
            r'故选：([ABCD])',
            r'故选([ABCD])',
            r'本题选([ABCD])',
            r'故([ABCD])正确',
            r'所以([ABCD])正确',
            r'([ABCD])项是正确答案',
            r'【答案】([ABCD])',
            r'选([ABCD])',
            r'是([ABCD])'
        ]

    for answer_pattern in answer_patterns:
        m = re.search(answer_pattern, gen_ans, re.M)
        if m:
            answer = m.group(1)
            return answer
    if gen_ans[0] in 'ABCD':
        return gen_ans[0]
    t = []
    for opt in 'ABCD':
        if opt in gen_ans:
            t.append(opt)
    if len(t) == 1:
        return t[0]
    return None


class ChatCW_Evaluator(Evaluator):
    def __init__(self,choices,k,model_name):
        super(ChatCW_Evaluator,self).__init__(choices,model_name,k)
        model_path="/workspace/yckj2257/0626/bloomz-7b-cw-padding-v16-0626"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half().cuda()
        self.model = self.model.eval()

    def format_example(self,line,include_answer=True,cot=False):
        example=line['question']
        for choice in self.choices:
            example+=f'\n{choice}. {line[f"{choice}"]}'

        #example+='\n答案：'
        if include_answer:
            if cot:
                ans=line["answer"]
                content="让我们一步一步思考，\n"+line["explanation"]+f"\n所以答案是{ans}。"
                return [
                    {"role":"user","content":example},
                    {"role":"assistant","content":content}
                ]
            else:
                return [
                    {"role":"user","content":example},
                    {"role":"assistant","content":line["answer"]}
                ]
        else:
            return [
                {"role":"user","content":example},
            ]

    def generate_few_shot_prompt(self,subject,dev_df,cot=False):
        #prompt=f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n"
        prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction:"
        k=self.k

        if self.k==-1:
            k=dev_df.shape[0]
        for i in range(k):
            tmp=self.format_example(dev_df.iloc[i,:],include_answer=True,cot=cot)
            '''
            if i==0:
                tmp[0]["content"]=f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n"+tmp[0]["content"]
            '''
            tmp[0]["content"]=f"以下是中国关于{subject}考试的单项选择题，请从'A','B','C','D'中选出其中的正确答案。\n"+tmp[0]["content"]
            user=tmp[0]['content']
            cwbot=tmp[1]['content']
            prompt+=f"\nHuman: {user}\nAssistant: {cwbot}\n"
        return prompt
    
    def gen_response(self,inputs):
        inputs_encode =self.tokenizer(inputs,return_tensors="pt",padding=True)
        for k in inputs_encode:
            inputs_encode[k]=inputs_encode[k].cuda()
        max_tok=2048-inputs_encode.input_ids.shape[1]
        outputs=self.model.generate(**inputs_encode,do_sample=False,repetition_penalty=1.0,max_new_tokens=max_tok, pad_token_id=2)
        input_len=torch.max(torch.sum(inputs_encode.attention_mask,axis=1))
        response = [
            self.tokenizer.decode(outputs[i][input_len:],skip_special_tokens=True)
            for i in range(outputs.shape[0])
        ]
        return response


    def eval_subject(self,subject_name,test_df,dev_df=None,few_shot=False,save_result_dir=None,cot=False):
        correct_num=0
        template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"

        if save_result_dir:
            result=[]
            score=[]
        if few_shot:
            few_shot_prompt=self.generate_few_shot_prompt(subject_name,dev_df,cot=cot)
        else:
            few_shot_prompt=f"{template}### Instruction:\n以下是中国关于{subject_name}考试的单项选择题，请从'A','B','C','D'中选出其中的正确答案。\n"
            #few_shot_prompt=f"{template}### Instruction:\n以下是中国关于{subject_name}考试的单项选择题，请从'A','B','C','D'四个类别中，选出正确的类别\n"
            #few_shot_prompt=f"{template}### Instruction:\n"

        answers=list(test_df['answer'])
        message_list=[]
        tar_list=[]
        for row_index,row in tqdm(test_df.iterrows(),total=len(test_df)):
            question=self.format_example(row,include_answer=False)
            #full_prompt=few_shot_prompt+"<|Human|>: "+question[0]['content']+" <eoh>\n<|MOSS|>:"
            full_prompt=few_shot_prompt+question[0]['content']+"\n### Response:"

            if few_shot:
                query = f"以下是中国关于{subject}考试的单项选择题，请从'A','B','C','D'中选出其中的正确答案。\n" + question[0]['content']
                full_prompt=few_shot_prompt + "\nHuman: {}\nAssistant: ".format(query) + "\n### Response:"

            #print(full_prompt)
            message_list.append(full_prompt)
            tar_list.append(answers[row_index])
            if len(message_list)%1==0 or row_index==len(test_df)-1:
                response_list = self.gen_response(message_list)
                for i,response_str in enumerate(response_list):
                    if get_ans(response_str):
                        if get_ans(response_str) == tar_list[i]:
                            correct_num+=1
                            correct=1
                        else:
                            correct=0
                    else:
                        round1 = question[0]['content']
                        round1_ans = response_str
                        round2_input = f"{template}### Instruction:\nHuman: {round1}\nAssistant: {round1_ans}" + "\nHuman: 请根据以上对话内容，输出A,B,C,D中的一个\nAssistant: \n### Response:"
                        round2_ans = self.gen_response([round2_input])[0]
                        response_str = round2_ans
                        print(response_str)
                        if get_ans(round2_ans):
                            if get_ans(round2_ans) == tar_list[i]:
                                correct_num+=1
                                correct=1
                            else:
                                correct = 0
                        else:
                            if round2_ans.strip()[0] == tar_list[i]:
                                correct_num+=1
                                correct=1
                            else:
                                correct=0 

                    if save_result_dir:
                        result.append(response_str)
                        score.append(correct)
                message_list=[]
                tar_list=[]

        correct_ratio=100*correct_num/len(answers)

        if save_result_dir:
            test_df['model_output']=result
            test_df["correctness"]=score
            test_df.to_csv(os.path.join(save_result_dir,f'{subject_name}_val.csv'),encoding="utf-8",index=False)
        return [len(answers), correct_num, correct_ratio]
