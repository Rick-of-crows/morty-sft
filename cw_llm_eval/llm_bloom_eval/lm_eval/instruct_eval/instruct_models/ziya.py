
from .base_model import BaseM

from transformers import AutoTokenizer
from transformers import LlamaForCausalLM
import torch

class Ziya(BaseM):
    def __init__(self, config):
        self.cfg = config
        model_dir = self.cfg["model_path"]
        self.device = torch.device("cuda")
        print("begin to load model:", model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = LlamaForCausalLM.from_pretrained(model_dir, device_map="auto")
        self.model.eval()
        self.max_new_tokens = self.cfg['max_length']
        self.do_sample = self.cfg['do_sample']
        self.generate_template = '<human>:{instruction}\n<bot>:'

    def init_chain_rank(self):
        pass

    def init_chain_score(self):
        pass

    def rank_generate(self, x, y1, y2):
        pass

    def score_generate(self, x, y):
        pass

    def base_generate(self, texts):
        assert len(texts)==1
        texts = [text['instruction'] for text in texts]
        text = texts[0]
        input_text = self.generate_template.format(instruction=text)
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        generate_ids = self.model.generate(input_ids, max_new_tokens=self.max_new_tokens, do_sample = self.do_sample, eos_token_id=2, bos_token_id=1, pad_token_id=0)
        result = self.tokenizer.batch_decode(generate_ids[:, input_ids.shape[-1]:])[0].strip().rstrip("</s>")
        #print("response:", result)
        return [result]