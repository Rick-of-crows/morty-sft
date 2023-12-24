
import openai
import os, sys
sys.path.append("..")
# import IPython
from langchain.llms import OpenAI
from dotenv import load_dotenv
import re
load_dotenv()

from .base_model import BaseM
from utils import *

openai.organization = os.getenv("ORGANIZATION")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")


from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

class ChatGPT(BaseM):
    def __init__(self, config):
        self.cfg = config
        assert "model_name" in config
        self.temperature = config["temperature"] if "temperature" in config else 0
        self.timeout = config["timeout"] if "timeout" in config else 120
        self.chat = ChatOpenAI(model_name=config["model_name"], temperature=self.temperature, request_timeout=self.timeout)
        #print("chat_info:", self.chat._default_params)
        self.init_generate_template()

    def init_generate_template(self):
        # self.generate_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \
        #     Instruction: {instruction} \
        #     Response:"
        self.generate_template = "{instruction}"
        if "generate_template" in self.cfg and 'instruction' in self.cfg["generate_template"]:
            self.generate_template = self.cfg["generate_template"]
        human_message_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
            template=self.generate_template,
            input_variables=["instruction"],
            )
        )
        chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
        self.base_chat = LLMChain(llm=self.chat, prompt=chat_prompt_template)

    def init_chain_rank(self):
        # 为了防止问题被截断，目前只支持比较两个答案
        template="You are a helpful assistant that compare two potential responses to an instruction."
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_template = '''Given the instruction: """
{instruction}
"""
and two potential responses:

Response A: """
{resp_1}
"""

Response B: """
{resp_2}
"""

Please evaluate the quality of the two responses, and then rank them in descending order, and finally output a ranking list. If two answers have similar or same quality, output the same rank like 1. Response A\n1. Response B.'''
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        self.chain_rank = LLMChain(llm=self.chat, prompt=chat_prompt)

    def construct_template(self, tpl):
        template = tpl['template']
        demo_input = tpl['demo_input']
        metric = tpl['metric']
        #demo_output = tpl['demo_output']
        demo_response = tpl['demo_response']
        std_output = tpl['std_output'] if 'std_output' in tpl else None
        if std_output is not None and std_output != "":
            if metric == "score":
                demo_output = tpl['demo_output']
                demo_info = "\n提供一个评判示例，问题:\"{}\"\n标准答案:\"{}\"\n模型回答:\"{}\"\n参考标准答案，对模型回答给出分数:\"{}\"".format(demo_input, std_output, demo_output, demo_response)
            elif metric == "rank":
                demo_output_a = tpl['demo_output_a']
                demo_output_b = tpl['demo_output_b']
                demo_info = "\n提供一个评判示例，问题:\"{}\"\n标准答案:\"{}\"\n答案A:\"{}\"\n答案B:\"{}\"\n通过参考标准答案，评判结果为:\"{}\"".format(demo_input, std_output, demo_output_a, demo_output_b, demo_response)
        else:
            if metric == "score":
                demo_output = tpl['demo_output']
                demo_info = "\n提供一个评判示例，问题:\"{}\"\n模型回答:\"{}\"\n按照要求，对模型回答给出分数:\"{}\"".format(demo_input, demo_output, demo_response)
            elif metric == "rank":
                demo_output_a = tpl['demo_output_a']
                demo_output_b = tpl['demo_output_b']
                demo_info = "\n提供一个评判示例:\n问题:\"{}\"\n答案A:\"{}\"\n答案B:\"{}\"\n按照要求，评判结果为:\"{}\"".format(demo_input, demo_output_a, demo_output_b, demo_response)
        return template + demo_info
        #return template

    def init_rank_by_category(self, tpl_path="instruct_models/rank_prompt.json"):
        tpl_dict = json_load(tpl_path)
        tpl_dict = prompt_dict_mapping(tpl_dict)
        self.chat_rank_tpl = {}
        for key in tpl_dict:
            tpl = tpl_dict[key]
            template = self.construct_template(tpl)
            system_message_prompt = SystemMessagePromptTemplate.from_template(template)
            ### no std output
            chat_openai = ChatOpenAI(model_name=self.cfg["model_name"], temperature=self.temperature, request_timeout=self.timeout)
            human_template = '''\n提供一个问题: "{instruction}"\n答案A:"{resp1}"\n答案B:"{resp2}"\n按照要求，评判结果为:'''
            human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
            chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
            self.chat_rank_tpl[key] = LLMChain(llm=chat_openai, prompt=chat_prompt)
            ### have std output
            chat_openai_std = ChatOpenAI(model_name=self.cfg["model_name"], temperature=self.temperature, request_timeout=self.timeout)
            human_template_std = '''\n提供一个问题: "{instruction}"\n标准答案:"{std_answer}"\n答案A:"{resp1}"\n答案B:"{resp2}"\n通过参考标准答案，评判结果为:'''
            human_message_prompt_std = HumanMessagePromptTemplate.from_template(human_template_std)
            chat_prompt_std = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt_std])
            #print("chat_prompt:", chat_prompt_std)
            self.chat_rank_tpl[key+"_std"] = LLMChain(llm=chat_openai_std, prompt=chat_prompt_std)

    def init_score_by_category(self, tpl_path="instruct_models/score_prompt.json"):
        tpl_dict = json_load(tpl_path)
        tpl_dict = prompt_dict_mapping(tpl_dict)
        #print("tpl_dict:", tpl_dict)
        self.chat_score_tpl = {}
        for key in tpl_dict:
            tpl = tpl_dict[key]
            template = self.construct_template(tpl)
            #template = tpl['template']
            #print("template:", template)
            system_message_prompt = SystemMessagePromptTemplate.from_template(template)
            ### no std output
            chat_openai = ChatOpenAI(model_name=self.cfg["model_name"], temperature=self.temperature, request_timeout=self.timeout)
            human_template = '''\n提供一个问题: "{instruction}" 和模型回答:"{resp}", 根据要求给出分数:'''
            human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
            chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
            self.chat_score_tpl[key] = LLMChain(llm=chat_openai, prompt=chat_prompt)
            ### have std output
            chat_openai_std = ChatOpenAI(model_name=self.cfg["model_name"], temperature=self.temperature, request_timeout=self.timeout)
            human_template_std = '''\n提供一个问题: "{instruction}"\n标准答案: "{std_answer}"\n模型回答: "{resp}"\n通过参考标准答案，对模型回答给出分数:'''
            human_message_prompt_std = HumanMessagePromptTemplate.from_template(human_template_std)
            chat_prompt_std = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt_std])
            #print("chat_prompt_std:", chat_prompt_std)
            self.chat_score_tpl[key+"_std"] = LLMChain(llm=chat_openai_std, prompt=chat_prompt_std)

    def init_chain_score(self):
        template="You are a helpful assistant that rate a potential response to an instruction that describes a task."
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_template = '''Given the instruction: """
{instruction}
"""
and a potential response: """
{resp}
"""

Please rate the quality of the responses and give a score ranging from 1 to 5. If the response does not meet the requirements of the instruction, the score is 1.

The score is'''
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        self.chain_score = LLMChain(llm=self.chat, prompt=chat_prompt)

    def init_chain_score_pair(self):
        template="You are a helpful and precise assistant for checking the quality of the answer."
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_template = "[Question]\n{instruction}\n\n[Assistant 1]\n{resp_1}\n\n[End of Assistant 1]\n\n[Assistant 2]\n{resp_2}\n\n[End of Assistant 2]\n\n \
We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 0 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        self.chain_score_pair = LLMChain(llm=self.chat, prompt=chat_prompt)

    def rank_old_generate(self, x, y1, y2, std=None, category=None):
        result_full = self.chain_rank.run(instruction=x, resp_1=y1, resp_2=y2)
        #pattern = r"[1-2]\.\s+(?:Response)?\s+([A-B])"
        pattern = r"([1-2])\.\s+(?:Response)?\s+([A-B])+\s+([1-2])\.\s+(?:Response)?\s+([A-B])"
        m = re.search(pattern, result_full)
        #print("result:", result_full)
        if m:
            if m.group(1) == m.group(3): ### tied
                return -1, result_full
            return ord(m.group(2))-ord('A'),result_full
        else:
            ### match first rank
            pattern = r"[1-2]\.\s+(?:Response)?\s+([A-B])"
            m2 = re.search(pattern, result_full)
            if m2:
                return ord(m2.group(1))-ord('A'),result_full
            else:
                return None,result_full

    def rank_generate(self, x, y1, y2, std, category):
        if category not in self.chat_rank_tpl:
            raise RuntimeError('Unknown category:%s'% category)
        if std != "":
            result_full = self.chat_rank_tpl[category+"_std"].run(instruction=x, std_answer=std, resp1=y1, resp2=y2)
        else:
            result_full = self.chat_rank_tpl[category].run(instruction=x, resp1=y1, resp2=y2)
        #print("result:", result_full)
        #pattern = r"([1-3])\.\s?([A-B])\s([1-3])\.\s?([A-B])"
        pattern = r"答案:\s*([A-B])([A-B])?"
        m = re.search(pattern, result_full)
        if m:
            if m.group(2) is not None:
                return -1, result_full
            return ord(m.group(1))-ord('A'),result_full
        else:
            return None,result_full

    def score_generate(self, x, y, std, category):
        if category not in self.chat_score_tpl:
            raise RuntimeError('Unknown category:%s'% category)
        if std != "":
            #print("instruction:", x, " std_answer:", std, " resp:", y)
            result_full = self.chat_score_tpl[category+"_std"].run(instruction=x, std_answer=std, resp=y)
        else:
            result_full = self.chat_score_tpl[category].run(instruction=x, resp=y)
        #print("instruct:", x)
        #print("response:", y)
        #print("result:",result_full)
        pattern = r"得分[:：]\s*(\d+(\.\d+)?)"
        m = re.search(pattern, result_full)
        if m:
            return m[1], result_full
        else:
            return None, result_full

    def score_pair_generate(self, x, y1, y2):
        result_full = self.chain_score_pair.run(instruction=x, resp_1=y1, resp_2=y2)
        #print("result_full:", result_full)
        pattern = r'\d+(\.\d+)?\s\d+(\.\d+)?'
        m = re.match(pattern, result_full)
        if m:
            #vals = m[0].split(" ")
            return m[0], result_full
        else:
            return None,result_full

    def base_generate(self, texts):
        assert(len(texts) == 1)
        texts = [text['instruction'] for text in texts]
        text = texts[0]
        #print("text:", text)
        try:
          #result = self.base_chat.run(text)
          result = self.base_chat.run(instruction=text)
        except Exception:
          print("Timeout:", text)
          result = ""
        #print("result:", result)
        return [result]