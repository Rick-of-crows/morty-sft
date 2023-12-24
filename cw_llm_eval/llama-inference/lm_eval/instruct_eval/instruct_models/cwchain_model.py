import re
from typing import Optional, List, Any, Mapping
from langchain.llms.base import LLM
from .base_model import BaseM
from .__init__ import *
from cwchain.agent.cw_agent import CwChain
from cwchain.models.llama import Llama

class EvalLM(LLM):
    model: BaseM
    @property
    def _llm_type(self) -> str:
        return "eval_lm"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        print(f"**************prompt**************\n{prompt}*********************************")
        items = [{'instruction':prompt, 'history':''}]
        output = self.model.base_generate(items)[0]
        if stop is not None:
            for s in stop:
                i = output.find(s)
                if i != -1:
                    output = output[:i]
        print(f"**************output**************\n{output}\n*********************************")
        return output
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model": self.model}
    
class CwChainModel(BaseM):
    def __init__(self, config):
        # TODO: support different model testing
        model = get_model(config['llm']['type'], config['llm']['config'])
        llm = EvalLM(model=model)
        # llm = Llama(proxy="", addr = config['work_addr'],
        #     port=config['work_port'],
        #     temperature = 0.001,
        #     max_new_tokens = 2048)
        self.agent = CwChain(config['tool_names'], llm)
        
    def rank_generate(self, x, y1, y2):
        pass

    def score_generate(self, x, y):
        pass
    
    def init_chain_rank(self):
        pass

    def init_chain_score(self):
        pass

    def base_generate(self, items):
        outputs = []
        for item in items:
            # intermediate includes a whole conversation
            output, intermediate = self.agent.run(item['instruction'], history = [item['history']])
            intermediate = intermediate.split("</Human>")[-1].strip()
            print(f"output = {intermediate}")
            outputs.append(intermediate)
        return outputs

