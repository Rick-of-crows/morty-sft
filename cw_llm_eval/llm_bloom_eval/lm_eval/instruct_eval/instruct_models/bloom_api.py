from .base_model import BaseM

import requests, json

class BloomApi(BaseM):
    def __init__(self, config):
        self.cfg = config
        self.init_generate_template()
        self.headers = {"Content-Type": "application/json"}
        self.web = 'http://{}:{}/generate'.format(self.cfg["addr"], self.cfg["port"])
        self.data = {
            "tokens_to_generate": 512,
            "do_sample":False,
            "bos_token_id": 1,
            "eos_token_id":2,
            "pad_token_id":3,
            "repetition_penalty":1.1,
        }

    def init_generate_template(self):
        self.generate_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction:\n{instruction}\n### Response:"
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
        texts = [text['instruction'] for text in texts]
        text = texts[0]
        input_text = self.generate_template.format(instruction=text)
        self.data["text"] = [input_text]
        response = ""
        try:
            responses = requests.request("POST", self.web, data=json.dumps(self.data), headers=self.headers, timeout=120)
            if responses.status_code == 200:
                response = responses.json()["text"][0].strip("\n")
            else:
                print("error code: ", responses.status_code, " instruction:", input_text)
        except requests.exceptions.Timeout:
            print("time out:", input_text)
        except requests.exceptions.ConnectionError:
            print("connection error:", input_text)
        
        return [response]