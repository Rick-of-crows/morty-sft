# -*- coding: utf-8 -*-
"""
# @Author :  yckj3008
# @Email  :  pengjinhua@cloudwalk.com
# @Time   :  2023/6/5 16:05
# @File   :  maas_engine.py
"""
from .base_model import BaseM
import requests
import json

class MaasEngine(BaseM):
    def __init__(self, config):
        self.cfg = config
        self.url = "{}/llm.inference/chat/completions".format(self.cfg["url"])
        self.init_generate_template()
        self.data = {
            "id": "pjhtest",
            "max_tokens": 3000,
            "messages": [
                {
                    "content": "",
                    "role": "user"
                }
            ],
            "model": "cloudwalk-turbo",
            "n": 1,
            "stream": False,
            "temperature": 1,
            "debug": False,
            "stop": ""
        }
        self.headers = {
            "Content-Type": "application/json"
        }

    def init_generate_template(self):
        self.generate_template = "User:{instruction}\n\nAssistant:"
        if "generate_template" in self.cfg and 'instruction' in self.cfg["generate_template"]:
            self.generate_template = self.cfg["generate_template"]

    def base_generate(self, texts):
        assert len(texts) == 1
        texts = [text['instruction'] for text in texts]
        print("texts:", texts[0])
        self.data["messages"][0]["content"] = texts[0]
        responses = requests.request("POST", self.url, headers=self.headers, data=json.dumps(self.data), timeout=300)
        if responses.status_code == 200 and responses.json()["choices"] != [] and responses.json()["choices"] is not None:
            response_text = responses.json()["choices"][0]["text"]
            #print("maas response_text is:", response_text)
        else:
            print("maas error code is:", responses.status_code)
            response_text = f"请求失败，错误码{responses.status_code}"
        return [response_text]

    def init_chain_rank(self):
        pass

    def init_chain_score(self):
        pass

    def rank_generate(self, x, y1, y2):
        pass

    def score_generate(self, x, y):
        pass
