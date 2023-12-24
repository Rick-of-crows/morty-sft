import torch
from .base_model import BaseM

import requests, json

class LLamaApi(BaseM):
    def __init__(self, config):
        self.cfg = config
        self.web = 'http://{}:{}/generate'.format(self.cfg["addr"], self.cfg["port"])
        self.proxy = self.cfg["proxy"] if "proxy" in self.cfg else ""
        self.proxies = {
            'http': 'socks5://' + self.proxy,
            'https': 'socks5://' + self.proxy
        }
        self.max_new_tokens = self.cfg["max_length"]
        self.headers = {"Content-Type": "application/json"}
        self.stop = ["### Human", "\n<Observation>", "\n<Human>","<Observation>","<Human>"]
        self.data = {
            "temperature": 1.0,
            "tokens_to_generate": self.max_new_tokens,
            "add_BOS": False,
            "top_k": 0,
            "top_p": 0.95,
            "greedy": True,
            "all_probs": False,
            "repetition_penalty": 1.0,
            "min_tokens_to_generate": 1,
            "end_strings": self.stop,
            "echo": False
        }

    def init_chain_rank(self):
        pass

    def init_chain_score(self):
        pass

    def rank_generate(self, x, y1, y2):
        pass

    def score_generate(self, x, y):
        pass

    def base_generate(self, items):
        self.data["sentences"] = [item['instruction'] for item in items]
        responses = requests.put(self.web, data=json.dumps(self.data), headers=self.headers)
        sentences = responses.json()['sentences']
        return sentences

