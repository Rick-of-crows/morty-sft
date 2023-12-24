from transformers import AutoTokenizer, AutoModelForCausalLM
import os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir,os.path.pardir,os.path.pardir)))
from lm_eval.models.gpt2 import GPT2LM
from lm_eval import evaluator, tasks, utils
from lm_eval.base import CacheHook
from lm_eval.tasks import ALL_TASKS
from tqdm import tqdm
import torch
import time
import json
import torch.nn.functional as F

class EvalHarnessAdaptor(GPT2LM):
    def __init__(self, model_dir, args):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        print("tokenizer load done...")
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
        print("model load done...")
        self.model.eval()
        self.is_main = True
        self.args = args

        self._max_length = args.max_seq_len
        self._batch_size = args.max_batch_size
        self._device = torch.device(f"cuda:{args.device_id}")

        self.cache_hook = CacheHook(None)

        #self.EOT_TOKEN_ID = self.generation_config.eos_token_id
        self.generate_max_len = self.args.max_generate_tokens

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def max_length(self):
        return self._max_length

    @property
    def device(self):
        return self._device

    def _model_call(self, inps):
        #print("inps:", inps.shape)
        with torch.no_grad():
            return self.model(inps)[0]

    def _model_generate(self, context, max_length, eos_token_id):
        #print("context:", context)
        '''return self.model.generate(
            context, max_length=max_length, do_sample=False
        )'''
        return self.model.generate(context, max_new_tokens=self.generate_max_len, do_sample=False)


def pad_batch(batch, pad_id, max_len):
    context_lengths = []
    max_context_length = max([len(tokens) for tokens in batch])
    for tokens in batch:
        context_length = len(tokens)
        if context_length < max_context_length + max_len:
            tokens.extend([pad_id] * (max_context_length + max_len - context_length))
        context_lengths.append(context_length)
    return batch, context_lengths

if __name__=='__main__':
    parser = argparse.ArgumentParser("lm_eval for transformers")
    parser.add_argument("-model_dir", help="transformer model checkpoint dir", required=True)
    parser.add_argument("-num_fewshot", help="fewshot num", type=int, default=0, required=False)
    parser.add_argument("-device_id", help="fewshot num", type=int, default=0, required=False)
    parser.add_argument("-task_list", help="eval task name list", nargs="+", required=True)
    parser.add_argument("-max_generate_tokens", help="max generate tokens num", type=int, default=32, required=False)
    parser.add_argument("-max_seq_len", help="max seq len", type=int, default=1024, required=False)
    parser.add_argument("-max_batch_size", help="max batch size", type=int, default=4, required=False)
    parser.add_argument("-ratio", help="ratio of datasets", type=float, default=1.0, required=False)
    args = parser.parse_args()
    start = time.time()
    assert(len(args.task_list) > 0), "Not support empty task_list."
    task_names = utils.pattern_match(args.task_list, tasks.ALL_TASKS)
    print("match tasknames:", task_names)
    adaptor = EvalHarnessAdaptor(args.model_dir, args)
    task_dict = tasks.get_task_dict(task_names, adaptor)
    results = evaluator.evaluate(adaptor, task_dict, False, args.num_fewshot, None, ratio=args.ratio)
    end = time.time()
    print("result:", results)
    print("evaluation of {} ends in {:.2f} sec, or {:.2f} min, or {:.2f} hr".format(args.task_list, end-start, (end-start)/60.0, (end-start)/3600.0))
