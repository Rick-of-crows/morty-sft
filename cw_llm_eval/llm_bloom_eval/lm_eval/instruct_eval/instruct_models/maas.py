# -*- coding: utf-8 -*-
"""
# @Author :  yckj3008
# @Email  :  pengjinhua@cloudwalk.com
# @Time   :  2023/6/2 14:05
# @File   :  maas.py
"""
from .base_model import BaseM
import requests
import json

class Maas(BaseM):
    def __init__(self, config):
        self.cfg = config
        self.url = "{}/api/gpt/completions".format(self.cfg["url"])
        #self.url = "{}/api/gpt/completions".format(self.cfg["url"])
        self.init_generate_template()
        self.data = {
            "model": "model",
            "messages": [{
                "role": "Human",
                "content": ""
            }]
        }
        self.headers = {
            "Content-Type": "application/json",
            "maasKey": self.cfg["maasKey"]
        }

    def init_generate_template(self):
        self.generate_template = "User:{instruction}\n\nAssistant:"
        if "generate_template" in self.cfg and 'instruction' in self.cfg["generate_template"]:
            self.generate_template = self.cfg["generate_template"]

    def base_generate(self, texts):
        assert len(texts) == 1
        texts = [text['instruction'] for text in texts]
        self.data["messages"][0]["content"] = texts[0]
        responses = requests.request("POST", self.url, headers=self.headers, data=json.dumps(self.data))
        if responses.status_code == 200 and responses.json()["content"] != "" and responses.json()["content"] is not None:
            response_content = responses.json()["content"]
            #print("maas response_content is:", response_content)
        else:
            print("maas error code is:", responses.status_code)
            response_content = f"请求失败，错误码{responses.status_code}"
        return [response_content]

    def init_chain_rank(self):
        pass

    def init_chain_score(self):
        pass

    def rank_generate(self, x, y1, y2):
        pass

    def score_generate(self, x, y):
        pass
