from .base_model import BaseM
import requests
import json

class ShuYan(BaseM):
    def __init__(self, config):
        self.cfg = config
        #self.web = 'http://{}:{}/generate'.format(self.cfg["addr"], self.cfg["port"])
        self.web = 'http://{}:{}/chat'.format(self.cfg["addr"], self.cfg["port"])
        self.max_new_tokens = self.cfg["max_length"]
        self.init_generate_template()
        '''self.data = {
            "inputs": "",
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "stop": ["</s>"]
            }
        }'''
        self.data = {
            "messages": [{"role":"user","content":""}]
        }
        self.headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'Apifox/1.0.0 (https://www.apifox.cn)',
            'access_token': 'U2FsdGVkX1/86tZ3izhnxFmUAap/QhVfHPIXOp32KIs=',
            'Accept': '*/*',
            'Host': '58.144.147.143:15101',
            'Connection': 'keep-alive',
        }

    def init_generate_template(self):
        self.generate_template = "User:{instruction}\n\nAssistant:"
        if "generate_template" in self.cfg and 'instruction' in self.cfg["generate_template"]:
            self.generate_template = self.cfg["generate_template"]

    def init_chain_rank(self):
        pass

    def init_chain_score(self):
        pass

    def rank_generate(self, x, y1, y2):
        pass

    def score_generate(self, x, y):
        pass

    def base_generate(self, texts):
        assert len(texts) == 1
        texts = [text['instruction'] for text in texts]
        #input_text = self.generate_template.format(instruction=texts[0])
        #self.data["inputs"] = input_text
        self.data["messages"][0]["content"] = texts[0]
        #print("web:", self.web, "data:", self.data)
        #responses = requests.put(self.web, data=json.dumps(self.data), headers=self.headers)
        #print("texts:", texts)
        responses = requests.request("POST", self.web, headers=self.headers, data=json.dumps(self.data))
        if responses.status_code == 200:
            #print("responses:", responses.json())
            response = responses.json()["generated_text"].rstrip("</s>")
            print("response:", response)
        else:
            print("error code:", responses.status_code)
            response = ""
        return [response]