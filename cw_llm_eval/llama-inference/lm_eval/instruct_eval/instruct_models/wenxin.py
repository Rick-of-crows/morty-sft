from .base_model import BaseM
import requests
import json


class WenXin(BaseM):
    def __init__(self, config):
        self.cfg = config
        self.url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token=" + self.get_access_token()
        self.data = {
            "messages": [{"role":"user","content":""}]
        }
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

    def get_access_token(self):
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {"grant_type": "client_credentials", "client_id": self.cfg['ak'], "client_secret": self.cfg['sk']}
        return str(requests.post(url, params=params).json().get("access_token"))

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
        text = texts[0]
        self.data["messages"][0]["content"] = text
        response = requests.request("POST", self.url, headers=self.headers, data=json.dumps(self.data))
        if response.status_code == 200:
            resp_dict = response.json()
            if "result" in resp_dict:
                return [response.json()["result"]]
            else:
                print("text:", text)
                print("response.json():", response.json())
                return [""]