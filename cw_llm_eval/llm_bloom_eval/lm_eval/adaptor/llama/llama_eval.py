from typing import Tuple
import os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir,os.path.pardir,os.path.pardir)))

import torch
import time
import json
from pathlib import Path

from lm_eval.models.gpt2 import GPT2LM
from lm_eval import evaluator, tasks, utils
from lm_eval.base import CacheHook
from lm_eval.tasks import ALL_TASKS
from tqdm import tqdm
import torch.nn.functional as F

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA

class EvalHarnessAdaptor(GPT2LM):
    def __init__(self, llama_model, args):
        self.llama_model = llama_model
        #self.model = self.llama_model.model
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", -1))
        self.is_main = local_rank == 0
        self._device = torch.device(f"cuda:{local_rank}")
        print("local_rank:", local_rank)
        self.tokenizer = llama_model.tokenizer
        self.args = args
        self.VOCAB_SIZE = self.tokenizer.n_words
        self.EOT_TOKEN_ID = self.tokenizer.eos_id
        #self.model = Float16Module(module=self.model, precision=cfg.trainer.precision)

        self._max_length = args.max_seq_len
        # For ds we split into mini batches and then micro batches to keep pipelining api happy.
        # With Megatron we just go to micro_batches directly
        self._batch_size = args.max_batch_size

        self.cache_hook = CacheHook(None)

        self.generate_max_len = self.args.max_generate_tokens

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, bos: bool, eos: bool):
        #return self.tokenizer.text_to_ids(string)
        return self.tokenizer.encode(string, bos, eos)

    def tok_decode(self, tokens):
        #return self.tokenizer.ids_to_text(tokens)
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in requests:
            if context == "":
                # end of text as context
                context_enc = [self.EOT_TOKEN_ID]
            else:
                context_enc = self.tok_encode(context, True, False)
                #print("cont:", context, " len:", context_enc)

            continuation_enc = self.tok_encode(continuation.strip(), False, False)
            ### add context_enc

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests):
        # TODO: Implement caching once we've confirmed the perplexity implementation
        # TODO: automatic batch size detection for vectorization
        loglikelihoods = []
        with torch.no_grad():
            for string, in tqdm(requests):
                rolling_token_windows = list(map(utils.make_disjoint_window, utils.get_rolling_token_windows(
                    token_list=self.tok_encode(string, True, False),
                    prefix_token=self.EOT_TOKEN_ID,
                    max_seq_len=self._max_length,
                    context_len=1,
                )))

                rolling_token_windows = [(None,) + x for x in rolling_token_windows]

                # TODO: extract out this call so it only gets called once and also somehow figure out partial caching for that
                string_nll = self._loglikelihood_tokens(rolling_token_windows, disable_tqdm=True)

                # discard is_greedy
                string_nll = [x[0] for x in string_nll]

                string_nll = sum(string_nll)
                loglikelihoods.append(string_nll)

        return loglikelihoods

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        """
        In this method, the model doesn't do any generation, but just returns log likelihoods
        for the next token, which eval harness uses to evaluate.
        :param requests: Dictionary of requests containing the context and the expected continuation.
        :param disable_tqdm: If True, disable tqdm progress bar.
        """

        disable_tqdm = disable_tqdm if self.is_main else True
        res = []
        res_len = 0  # storing the result length for later
        self.llama_model.model.eval()
        with torch.no_grad():

            def _collate(x):
                toks = x[1] + x[2]
                return (-len(toks), tuple(toks))

            reord = utils.Reorderer(requests, _collate)
            for chunk in utils.chunks(
                tqdm(reord.get_reordered(), disable=disable_tqdm), self.batch_size
            ):
                inps, contlens, inplens, padding_length = [], [], [], None
                for cache_key, context_enc, continuation_enc in chunk:
                    # when too long to fit in context, truncate from the left
                    #print("cc:", len(context_enc), len(continuation_enc), self.max_length)
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self._max_length + 1) :][:-1],
                        dtype=torch.long,
                    ).to(self.device)
                    (inplen,) = inp.shape

                    cont = continuation_enc

                    # since in _collate we make sure length is descending, the longest is always the first one.
                    padding_length = (
                        padding_length if padding_length is not None else inplen
                    )

                    # pad to length
                    inp = torch.cat(
                        [
                            inp,  # [seq]
                            torch.zeros(padding_length - inplen, dtype=torch.long).to(
                                inp.device
                            ),  # [padding_length - seq]
                        ],
                        dim=0,
                    )
                    inps.append(inp.unsqueeze(0))
                    #print("inp:", inp.shape)
                    contlens.append(cont)
                    inplens.append(inplen)

                logits = self._model_call(torch.cat(inps, dim=0))
                res_len += len(chunk)

                if logits is not None:
                    multi_logits = F.log_softmax(logits, dim=-1)  # [batch, seq, vocab]
                    for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(
                        chunk, multi_logits, inps, inplens, contlens
                    ):
                        contlen = len(cont_toks)
                        logit_debug = logits.argmax(dim=-1)
                        #print("logits_v1:", logits.shape)
                        logits = logits[inplen - contlen: inplen].unsqueeze(0)
                        greedy_tokens = logits.argmax(dim=-1)
                        # cont_toks :: [1, seq]
                        cont_toks = (
                            torch.tensor(cont_toks, dtype=torch.long)
                            .unsqueeze(0)
                            .to(multi_logits.device)
                        )
                        #print("greedy_tokens:", greedy_tokens, " cont_toks:", cont_toks)
                        #print("greedy_tokens_decode:", self.tok_decode(greedy_tokens.tolist()[0]), " cont_toks_decode:", self.tok_decode(cont_toks.tolist()[0]))
                        max_equal = (greedy_tokens == cont_toks).all()
                        logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]
                        #print("shape:", logits.shape, " sum:", logits.sum())
                        answer = (float(logits.sum()), bool(max_equal))
                        #answer = (float(logits.mean()), bool(max_equal))

                        # partial caching
                        if cache_key is not None:
                            self.cache_hook.add_partial(
                                "loglikelihood", cache_key, answer
                            )

                        res.append(answer)

        return reord.get_original(res)
    
    @torch.inference_mode()
    def _model_call(self, inps):
        #inps = torch.ones_like(inps)
        #print("inputs:", inps[0][:5], " last:", inps[0][-5:])
        _bsz, seqlen = inps.shape
        #print("tok_inps:", inps.shape, inps[-1][-5:])
        h = self.llama_model.model.tok_embeddings(inps)
        #print("tok_embed:", h.shape, h[0][0][-3:], " last:", h[0][-1][-3:])  # (1, 111, 4096)
        freqs_cis = self.llama_model.model.freqs_cis.to(h.device)
        freqs = freqs_cis[:seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=inps.device)
            mask = torch.triu(mask, diagonal=1).type_as(h)
        #print("len:", len(self.llama_model.model.layers))
        for layer in self.llama_model.model.layers:
            h = layer(h, 0, freqs, mask)
        h = self.llama_model.model.norm(h)
        #print("norm:", h.shape, h[0][0][-3:], " last:", h[0][-1][-3:])
        output = self.llama_model.model.output(h)
        return output

    def greedy_until(self, requests):
        res = []
        def _collate(x):
            toks = self.tok_encode(x[0], False, False)
            return (-len(toks), x[0])

        reord = utils.Reorderer(requests, _collate)

        for chunk in utils.chunks(
            tqdm(reord.get_reordered()), self.batch_size
        ):
            #print("chunk:", len(chunk))
            inps, untils = [], []
            for context, until in chunk:
                #context = "I believe the meaning of life is simply put, the theory of relativity states that Building a website can be done"    
                enc = self.tok_encode(context, True, False)
                if len(enc) + self.generate_max_len > self._max_length:
                    new_enc = enc[-(self._max_length - self.generate_max_len):]
                    context = self.tok_decode(new_enc)
                inps.append(context)
                untils.append(until)

            response = self.llama_model.generate(inps, max_gen_len=self.generate_max_len, temperature=0, top_p=1.0)
            #print("response:", response, len(response))
            
            for idx in range(len(response)):
                #print("response:", response[idx])
                s = response[idx][len(inps[idx]):].strip()
                #print("sssssss:", response[idx], " ttttttt:", s)
                for term in untils[idx]:
                    s = s.split(term)[0]
                #print("idx:", idx, " response:", s)
                self.cache_hook.add_partial("greedy_until", (inps[idx], untils[idx]), s)
                res.append(s)
        return reord.get_original(res)

def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

if __name__=='__main__':
    parser = argparse.ArgumentParser("lm_eval for llama")
    parser.add_argument("-ckpt_dir", help="llama model checkpoint dir", required=True)
    parser.add_argument("-tokenizer_path", help="llama tokenizer model path", required=True)
    parser.add_argument("-num_fewshot", help="fewshot num", type=int, default=0, required=False)
    parser.add_argument("-task_list", help="eval task name list", nargs="+", required=True)
    parser.add_argument("-max_generate_tokens", help="max generate tokens num", type=int, default=32, required=False)
    parser.add_argument("-max_seq_len", help="max seq len", type=int, default=1024, required=False)
    parser.add_argument("-max_batch_size", help="max batch size", type=int, default=4, required=False)
    args = parser.parse_args()
    start = time.time()
    assert(len(args.task_list) > 0), "Not support empty task_list."
    local_rank, world_size = setup_model_parallel()
    task_dict = tasks.get_task_dict(args.task_list)

    ckpt_dir = args.ckpt_dir
    #print(args.ckpt_dir, args.tokenizer_path, local_rank, world_size, args.max_seq_len, args.max_batch_size)
    llama_model = load(args.ckpt_dir, args.tokenizer_path, local_rank, world_size, args.max_seq_len, args.max_batch_size)

    print("load done...")

    adaptor = EvalHarnessAdaptor(llama_model, args)
    results = evaluator.evaluate(adaptor, task_dict, False, args.num_fewshot, None)
    end = time.time()
    print("result:", results)
    print("evaluation of {} ends in {:.2f} sec, or {:.2f} min, or {:.2f} hr".format(args.task_list, end-start, (end-start)/60.0, (end-start)/3600.0))

    

    '''prompts = ["Tell me an interesting fact about space travel."]
    results = generator.generate(prompts, max_gen_len=args.max_generate_tokens, temperature=0, top_p=1.0)
    for result in results:
        print(result)
        print("\n==================================\n")'''