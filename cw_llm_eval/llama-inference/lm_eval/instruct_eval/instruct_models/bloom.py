import os
import re

from transformers import AutoTokenizer, AutoModelForCausalLM
from .base_model import BaseM


class BLOOM(BaseM):
    def __init__(self, config):
        self.cfg = config
        model_dir = self.cfg["model_path"]
        print("begin to load model:", model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
        print("model load done...")
        self.model.eval()
        self.max_new_tokens = self.cfg['max_length']
        self.do_sample = self.cfg['do_sample']
        ## init rank and score
        #self.init_chain_rank()
        #self.init_chain_score()

    def init_chain_rank(self):
        pass

    def init_chain_score(self):
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
        inputs = self.tok_encode(text)
        inplen = len(inputs[0])
        outputs = self.model.generate(inputs, max_new_tokens = self.max_new_tokens, do_sample=self.do_sample)
        results = self.tok_decode(outputs[0][inplen:])
        return [results]

    def rank_generate(self, x, y1, y2):
        pass

    def score_generate(self, x, y):
        pass

if __name__ == '__main__':
    model_dir = "/nlp_data/zoo/pretrain/huggingface/bloom/bloomz-7b1/"
    #model_dir = "/nlp_data/zoo/pretrain/huggingface/bloom/bloom-560m/"
    config = {'max_tokens':128, 'do_sample':False}
    model = BLOOM(model_dir, config)
    model.init_chain_rank()
    model.init_chain_score()

    #model.init_chain_rank()
    instruction_example = '''如果我胃疼的话,去医院应该挂什么科室'''
    resp_1_example = '''挂急诊科'''
    resp_2_example = '''消化内科'''

    results = model.rank_generate(instruction_example, resp_1_example, resp_2_example)
    r1_results = model.score_generate(instruction_example, resp_1_example)
    r2_results = model.score_generate(instruction_example, resp_2_example)




