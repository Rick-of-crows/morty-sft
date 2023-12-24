import os
import re

from transformers import AutoTokenizer, AutoModelForCausalLM
from .base_model import BaseM
import torch

class VICUNA(BaseM):
    def __init__(self, config):
        self.cfg = config
        model_dir = self.cfg["model_path"]
        print("begin to load model:", model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()
        print("model load done...")
        self.init_generate_template()
        self.max_new_tokens = self.cfg['max_length']
        self.temperature = self.cfg["temperature"] if "temperature" in config else 0

    def init_generate_template(self):
        self.generate_template = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: {instruction} ###Assistant:"
        if "generate_template" in self.cfg and 'instruction' in self.cfg["generate_template"]:
            self.generate_template = self.cfg["generate_template"]

    def init_chain_rank(self):
        template="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: You are a helpful assistant that compare two potential responses to an instruction."
        human_template = "Given the instruction: {instruction} \
    and two potential responses: \n\nResponse A: {resp_1} \n\nResponse B: {resp_2}\
    Please evaluate the quality of the two responses, and then rank them in descending order. ###Assistant:"
        self.rank_template = template + human_template

    def init_chain_score(self):
        pass

    def rank_generate(self, x, y1, y2):
        prompt = self.rank_template.format(instruction=x, resp_1=y1, resp_2=y2)
        #print("prompt:", prompt)
        pre = 0
        for outputs in self.generate_stream(prompt):
            outputs = outputs[len(prompt) + 1:].strip()
            outputs = outputs.split(" ")
            now = len(outputs)
            if now - 1 > pre:
                pre = now - 1
        result = " ".join(outputs)
        #print("result:", result)
        return None, result

    def score_generate(self, x, y):
        pass

    def base_generate(self, texts):
        assert len(texts)==1
        texts = [text['instruction'] for text in texts]
        text = texts[0]
        prompt = self.generate_template.format(instruction=text)
        #print("prompt:", prompt)
        pre = 0
        for outputs in self.generate_stream(prompt):
            outputs = outputs[len(prompt) + 1:].strip()
            outputs = outputs.split(" ")
            now = len(outputs)
            if now - 1 > pre:
                pre = now - 1
        result = " ".join(outputs)
        #print("result:", result)
        return [result]
 
    @torch.inference_mode()
    def generate_stream(self, prompt, device=0,
                        context_len=2048, stream_interval=2):
        """Adapted from fastchat/serve/model_worker.py::generate_stream"""

        #prompt = params["prompt"]
        l_prompt = len(prompt)
        temperature = self.temperature
        max_new_tokens = self.max_new_tokens
        #stop_str = params.get("stop", None)
        stop_str = "###"

        input_ids = self.tokenizer(prompt).input_ids
        output_ids = list(input_ids)

        max_src_len = context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]

        for i in range(max_new_tokens):
            if i == 0:
                out = self.model(
                    torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                attention_mask = torch.ones(
                    1, past_key_values[0][0].shape[-2] + 1, device=device)
                out = self.model(input_ids=torch.as_tensor([[token]], device=device),
                            use_cache=True,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values)
                logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[0][-1]
            if temperature < 1e-4:
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(token)

            if token == self.tokenizer.eos_token_id:
                stopped = True
            else:
                stopped = False

            if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
                output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                pos = output.rfind(stop_str, l_prompt)
                if pos != -1:
                    output = output[:pos]
                    stopped = True
                yield output

            if stopped:
                break

        del past_key_values

