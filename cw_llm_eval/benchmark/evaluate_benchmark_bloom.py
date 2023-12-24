# coding=utf-8
import datetime
import os, sys, argparse
import json
import time
import torch
from tqdm import tqdm
from typing import Iterable
from transformers import AutoTokenizer, AutoModelForCausalLM
from tasks.clue import OCNLI, CMNLI, AFQMC, TNEWS, CMRC2018, DRCD
from tasks.dureader import DUREADER
from tasks.nlpcc import NLPCC
from lm_eval.base import BaseLM
from lm_eval import evaluator, tasks, utils
from lm_eval.base import CacheHook
from lm_eval.tasks import ALL_TASKS, TASK_REGISTRY

TASK_REGISTRY.update(
    {
        "ocnli": OCNLI,
        "cmnli": CMNLI,
        "afqmc": AFQMC,
        "tnews": TNEWS,
        "cmrc2018": CMRC2018,
        "drcd": DRCD,
        "dureader": DUREADER,
        "nlpcc": NLPCC,
    }
)

parser = argparse.ArgumentParser("lm-evaluate")
parser.add_argument("--model_name_or_path", help="transformer model name or path", required=True)
parser.add_argument("--tokenizer_name_or_path", help="transformer tokenizer name or path", required=True)
parser.add_argument("--batch_size", help="batch size of evaluate", type=int, default=4)
parser.add_argument("--max_seq_length", help="max seq length of input", type=int, default=1024)
parser.add_argument("--max_generate_tokens", help="number of max generate tokens", type=int, default=100)
parser.add_argument("--fp16", action="store_true", help="Run model in fp16 mode.")
parser.add_argument("--device_id", help="device id of gpu", type=str, default="0")
parser.add_argument("--num_fewshot", help="number of fewshot", type=int, default=0)
parser.add_argument("--task_list", type=str, default="all", help='Either "all" or comma separated list of tasks.')
parser.add_argument("--results_path", type=str, default=None, help="Path to where the results will be stored.")
parser.add_argument("--test_numbers", type=int, default=None, help="number of test samples")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id


class EvalHarnessAdaptor(BaseLM):
    def __init__(self, args):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._batch_size = args.batch_size

        self._max_length = args.max_seq_length

        self.generate_max_len = args.max_generate_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

        self.model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

        if args.fp16:
            self.model.half()

        self.model.to(self.device)

        self.model.eval()

        self.cache_hook = CacheHook(None)

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return self.generate_max_len

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_decode(self, tokens: Iterable[int]):
        return self.tokenizer.decode(tokens)

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps)[0]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, pad_token_id=eos_token_id, do_sample=False
        )


def main():
    start = time.time()

    adaptor = EvalHarnessAdaptor(args)

    task_list = ALL_TASKS if args.task_list == "all" else args.task_list.split(",")
    task_dict = tasks.get_task_dict(task_list)

    global_results = {"results": {}, "versions": {}}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    iteration_id = args.model_name_or_path.split("/")[-1].replace("/", "")
    results_path = args.results_path.replace(".json", f"_lm-eval_{iteration_id}_{timestamp}.json")
    # Backup file in case of interruption during writing
    results_path_backup = args.results_path.replace(".json", f"_lm-eval_{iteration_id}_{timestamp}_backup.json")
    for task_name, task in task_dict.items():
        if args.test_numbers == -1:
            args.test_numbers = None
        results = evaluator.evaluate(
            adaptor, {task_name: task}, False, args.num_fewshot, args.test_numbers, bootstrap_iters=2
        )
        global_results["results"] = {**global_results["results"], **results["results"]}
        global_results["versions"] = {**global_results["versions"], **results["versions"]}
        print(json.dumps(results, indent=2))
        with open(results_path, "w") as outfile:
            json.dump(global_results, outfile, indent=4)
        with open(results_path_backup, "w") as outfile:
            json.dump(global_results, outfile, indent=4)
    end = time.time()
    print("evaluation time of {} is {:.2f} sec".format(args.task_list, end - start))


if __name__ == "__main__":
    main()
