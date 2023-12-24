import os
import re

from .base_model import BaseM

import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor
from flagai.model.predictor.aquila import aquila_generate
from flagai.data.tokenizer import Tokenizer

class Aquila(BaseM):
    def __init__(self, config):
        self.cfg = config
        assert "model_dir" in config
        assert "model_name" in config
        self.init_generate_template()
        loader = AutoLoader("lm", model_dir=config["model_dir"], \
            model_name=config["model_name"], use_cache=True)
        print("get aquila model done...")
        self.model = loader.get_model().eval()
        self.tokenizer = loader.get_tokenizer()
        self.model.half().cuda()

    def init_generate_template(self):
        self.generate_template = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: {instruction}###Assistant:"
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
        tokens = self.tokenizer.encode_plus(input_text, None, max_length=None)['input_ids']
        tokens = tokens[1:-1]
        with torch.no_grad():
            out = aquila_generate(self.tokenizer, self.model, [text], max_gen_len:=384, temperature=0, prompts_tokens=[tokens])
            result = out[len(input_text):].replace("[UNK]","").strip()
        #print("result:", result)
        return [result]

        
