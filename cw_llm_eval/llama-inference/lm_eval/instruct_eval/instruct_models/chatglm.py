import os
import re

from transformers import AutoTokenizer, AutoModel
from .base_model import BaseM


class ChatGLM(BaseM):
    def __init__(self, config):
        self.cfg = config
        assert "model_path" in config
        model_dir = config['model_path']
        print("begin to load model:", model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
        print("model load done...")
        self.model.eval()

        self.max_new_tokens = self.cfg['max_length']
        self.do_sample = self.cfg['do_sample']

    def init_chain_rank(self):
        pass

    def init_chain_score(self):
        pass

    def rank_post_process(self, results):
        pass

    def score_post_process(self, results):
        pass

    def tok_encode(self, text):
        inputs = self.tokenizer.encode(text, return_tensors="pt").cuda()
        return inputs

    def tok_decode(self, inputs):
        return self.tokenizer.decode(inputs)

    def base_generate(self, texts):
        assert len(texts)==1
        texts = [text['instruction'] for text in texts]
        text = texts[0]
        result, history = self.model.chat(self.tokenizer, text, history=[])
        return [result]

    def rank_generate(self, x, y1, y2):
        pass

    def score_generate(self, x, y):
        pass

if __name__ == '__main__':
    model_dir = "/nlp_data/zoo/pretrain/glm_model/6B/"
    config = {'max_tokens':128, 'do_sample':False}
    model = ChatGLM(model_dir, config)
    model.init_chain_rank()
    model.init_chain_score()

    #instruction_example = '''如果我胃疼的话,去医院应该挂什么科室'''
    #resp_1_example = '''挂急诊科'''
    #resp_2_example = '''消化内科'''
    instruction_example = '''中国的首都在哪里?'''
    resp_1_example = '''北京'''
    resp_2_example = '''上海'''


    results = model.rank_generate(instruction_example, resp_1_example, resp_2_example)
    r1_results = model.score_generate(instruction_example, resp_1_example)
    r2_results = model.score_generate(instruction_example, resp_2_example)




