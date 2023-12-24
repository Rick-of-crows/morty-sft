# under the license https://huggingface.co/spaces/bigscience/license

from functools import reduce
from logging import logMultiprocessing
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir,os.path.pardir)))

from lm_eval.models.gpt2 import GPT2LM
from lm_eval import evaluator, tasks, utils
from lm_eval.base import CacheHook
from megatron.checkpointing import load_checkpoint
from tqdm import tqdm
import torch.nn.functional as F

from lm_eval.tasks import ALL_TASKS
from pretrain_gpt import model_provider
import numpy as np
import time

import torch
from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
#from megatron import mpu
from megatron.core import mpu, tensor_parallel
from megatron.training import setup_model_and_optimizer, get_model
#from megatron.mpu.mappings import gather_from_tensor_model_parallel_region
from megatron.model import GPTModel

from megatron.utils import get_ltor_masks_and_position_ids, unwrap_model
from megatron.p2p_communication import recv_forward, send_forward
import pickle
import json

'''from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.model.distributed import DistributedDataParallel as LocalDDP
from megatron.model.module import Float16Module'''

class EvalHarnessAdaptor(GPT2LM):
    def __init__(self, model, tokenizer):
        args = get_args()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.VOCAB_SIZE = tokenizer.vocab_size
        self.EOT_TOKEN_ID = tokenizer.eod

        self._max_length = args.seq_length

        # For ds we split into mini batches and then micro batches to keep pipelining api happy.
        # With Megatron we just go to micro_batches directly
        self._batch_size = args.micro_batch_size

        self.cache_hook = CacheHook(None)
        self.is_main = args.rank == 0
        self.is_local_main = args.local_rank == 0
        #self._device = get_accelerator().current_device_name()
        self._device = torch.device(f"cuda:{args.local_rank}")
        self.is_model_parallel = mpu.get_tensor_model_parallel_world_size() > 1
        self.is_pipe_parallel = mpu.get_pipeline_model_parallel_world_size() > 1
        self.dp_world_size = mpu.get_data_parallel_world_size()
        self.dp_rank = mpu.get_data_parallel_rank()
        self.adaptive_seq_len = False

        self.is_last_stage = True if not self.is_pipe_parallel else mpu.is_pipeline_last_stage()  # only the last stage of the pipeline model will receive the logits
        #self.is_last_stage = True

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.tokenizer.encode(string)

    def tok_decode(self, tokens):
        return self.tokenizer.tokenizer.decode(tokens)

    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in requests:
            '''print("context:", context)
            print("continuation:", continuation)'''
            if context == "":
                # end of text as context
                context_enc = [self.EOT_TOKEN_ID]
            else:
                context_enc = self.tok_encode(context)
                #print("cont:", context, " len:", context_enc)

            continuation_enc = self.tok_encode(continuation)
            #print("enc:", continuation_enc)
            #print("context:", context, " continuation:", continuation)
            #print("c_enc:", len(context_enc), " a_enc:", len(continuation_enc))
            #print("c_enc:", len(context_enc), " id:", self.EOT_TOKEN_ID, " max:", self._max_length)
            ### add context_enc
            '''num_pad = (self._max_length - len(context_enc))
            context_enc += [self.EOT_TOKEN_ID] * num_pad'''

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests):
        # TODO: Implement caching once we've confirmed the perplexity implementation
        # TODO: automatic batch size detection for vectorization

        loglikelihoods = []
        with torch.no_grad():
            for string, in tqdm(requests):
                rolling_token_windows = list(map(utils.make_disjoint_window, utils.get_rolling_token_windows(
                    token_list=self.tok_encode(string),
                    prefix_token=self.EOT_TOKEN_ID,
                    max_seq_len=self._max_length,
                    context_len=1,
                )))
                #print("string:", len(string))

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
        #self.model.module.inference_mode(
        #    use_cache=False
        #)  # tell model to gather parallel outputs, but not cache key-value pairs

        disable_tqdm = disable_tqdm if self.is_main else True
        res = []
        res_len = 0  # storing the result length for later
        '''print("req:", len(requests), requests[0])
        print("dd:", requests[0][1])
        print("de:", requests[0][2])'''
        self.model.eval()
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
                    #print("cont:", len(cont))

                    # since in _collate we make sure length is descending, the longest is always the first one.
                    padding_length = (
                        padding_length if padding_length is not None else inplen
                    )

                    # pad to length
                    #print("ii0:", inp.shape, padding_length, inplen)
                    inp = torch.cat(
                        [
                            inp,  # [seq]
                            torch.zeros(padding_length - inplen, dtype=torch.long).to(
                                inp.device
                            ),  # [padding_length - seq]
                        ],
                        dim=0,
                    )
                    #print("xxx:", inp[:5])
                    '''if inplen < self._max_length:
                      num_pad = (self._max_length - inplen)
                      pad_tensor = torch.tensor([self.EOT_TOKEN_ID] * num_pad, dtype=torch.long).to(self.device)
                      inp = torch.cat([inp, pad_tensor], dim=0)'''
                    inps.append(inp.unsqueeze(0))
                    #print("ii2:", len(inps))
                    contlens.append(cont)
                    inplens.append(inplen)

                logits = self._model_call(torch.cat(inps, dim=0))
                #print("logits:", logits.shape)
                res_len += len(chunk)

                if logits is not None:
                    multi_logits = F.log_softmax(logits, dim=-1)  # [batch, seq, vocab]
                    for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(
                        chunk, multi_logits, inps, inplens, contlens
                    ):
                        contlen = len(cont_toks)
                        logit_debug = logits.argmax(dim=-1)
                        #print("logits_v1:", logits.shape)
                        logits = logits[inplen - contlen : inplen].unsqueeze(0)
                        #logits = logits[inplen - contlen - 4 : inplen - 4].unsqueeze(0)  # [1, seq, vocab]
                        #logits = logits[inplen: inplen + contlen].unsqueeze(0)
                        #print("logits_v2:", logits.shape, inplen, contlen)
                        greedy_tokens = logits.argmax(dim=-1)
                        #print("debug:", logit_debug[inplen - contlen - 6:inplen], cont_toks)
                        #print("greedy:", greedy_tokens, cont_toks)
                        # cont_toks :: [1, seq]
                        cont_toks = (
                            torch.tensor(cont_toks, dtype=torch.long)
                            .unsqueeze(0)
                            .to(multi_logits.device)
                        )
                        max_equal = (greedy_tokens == cont_toks).all()
                        #print("greedy_tokens:", greedy_tokens, cont_toks.shape, max_equal)
                        logits = torch.gather(
                            logits, 2, cont_toks.unsqueeze(-1)
                        ).squeeze(
                            -1
                        )  # [1, seq]
                        #print("gather:", logits.shape)
                        answer = (float(logits.sum()), bool(max_equal))

                        # partial caching
                        if cache_key is not None:
                            self.cache_hook.add_partial(
                                "loglikelihood", cache_key, answer
                            )

                        res.append(answer)

            # broadcast results to all ranks
            if self.is_pipe_parallel:
                src_rank = self.model.grid.stage_to_global(self.model.num_stages - 1)
                if res:
                    logits_sums, max_equals = list(zip(*res))
                    logits_sums = torch.FloatTensor(logits_sums).cuda()
                    max_equals = torch.LongTensor(max_equals).cuda()
                else:
                    logits_sums = torch.zeros(res_len, dtype=torch.float32).cuda()
                    max_equals = torch.zeros(res_len, dtype=torch.int64).cuda()
                torch.distributed.broadcast(
                    tensor=logits_sums,
                    src=src_rank,
                    group=mpu.get_pipe_parallel_group(),
                )
                torch.distributed.broadcast(
                    tensor=max_equals, src=src_rank, group=mpu.get_pipe_parallel_group()
                )
                max_equals = [bool(i) for i in max_equals.tolist()]
                logits_sums = logits_sums.tolist()
                res = list(zip(logits_sums, max_equals))

        #self.model.module.train_mode()  # set back to train mode
        return reord.get_original(res)

    def create_model_inputs(self, tokens):
        args = get_args()

        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            self.EOT_TOKEN_ID,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)
        #tokens = torch.zeros_like(tokens)
        return (tokens, position_ids, attention_mask), (tokens, loss_mask)

    def _dp_scatter(self, inps):
        """
        Scatters the inputs to all data parallel ranks.
        """
        batch_size = inps.shape[0]
        padded = False
        if batch_size % self.dp_world_size != 0:
            # The last batch could potentially not fill the full batch size (if the dataset size is not divisible by batch size)
            # In this case we pad the batch
            padded_size = self.dp_world_size - (batch_size % self.dp_world_size)

            '''print_rank_0(
                f"WARNING: Batch size ({batch_size}) must be divisible by dp world size ({self.dp_world_size}). Padding inputs to {padded_size}."
            )'''

            inps = torch.cat(
                [inps] + [inps[0:1, ...] for _ in range(padded_size)], dim=0
            )  # pad with first inp item
            padded = True

        assert (
            inps.shape[0] % self.dp_world_size == 0
        ), f"batch size ({inps.shape[0]}) must be divisible by dp world size ({self.dp_world_size})"

        # get a chunk for each data parallel rank
        chunk_size = inps.shape[0] // self.dp_world_size
        #print("chunk:", chunk_size, batch_size, padded)
        inps = inps[self.dp_rank * chunk_size : (self.dp_rank + 1) * chunk_size]
        # make a dummy dataloader / iterator to pass to model
        # we need to do this because deepspeed pipe parallel only takes an iterator
        # in this format
        #return iter([{"text": F.pad(inps, pad=(0, 1))}]), padded
        #print("input:", inps.shape, " pad:", F.pad(inps, pad=(0, 1)).shape)
        #return F.pad(inps, pad=(0, 1)), padded
        return inps, padded

    def _dp_gather(self, logits):
        """
        Gather logits from all data parallel ranks
        """
        if logits is not None:
            tensor_list = [torch.zeros_like(logits) for _ in range(self.dp_world_size)]
            torch.distributed.all_gather(
                tensor_list, logits, group=mpu.get_data_parallel_group()
            )
            logits = torch.cat(tensor_list, dim=0)
            return logits

    def _model_call(self, inps):
        args = get_args()
        # Since the shape of the micro-batch will change
        # We need set the correct shapes here
        # So that latter pipeline stages knows which shapes to expect.
        # Otherwise we will deadlock.

        '''args.micro_batch_size = len(inps)
        args.seq_length = len(inps[0])
        args.max_position_embeddings = args.seq_length'''
        #print("seq_len:", args.seq_length, args.max_position_embeddings)

        # scatter inputs to all dp ranks:
        inps, padded = self._dp_scatter(inps)

        '''input_tensor = recv_forward()
        # Forward pass through the model.
        unwrapped_model = unwrap_model(self.model, (torchDDP, LocalDDP, Float16Module))
        unwrapped_model.set_input_tensor(input_tensor)'''
        ## debug
        output = self.model(*self.create_model_inputs(inps)[0])
        #print(output[0,0,:5])
        #send_forward(output)

        # gather outputs from all dp ranks:
        output = self._dp_gather(output)

        '''if mpu.is_pipeline_last_stage():
            return tensor_parallel.gather_from_tensor_model_parallel_region(output)[..., :self.tokenizer.vocab_size]
        else:
            return None'''
        return output


from megatron.initialize import initialize_megatron
import megatron

def override_args(args, override_args, skip_keys, skip_if_specified_keys):
    for k, v in vars(override_args).items():
        if k in skip_keys:
            continue
        if k in skip_if_specified_keys and getattr(args, k) is not None:
            continue
        setattr(args, k, v)

def tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='Evaluation options')
    group.add_argument('--task_list', type=str, default = "all", help='Either "all" or comma separated list of tasks.')
    group.add_argument('--results_path', type=str, default = "./results.json", help='Path to where the results will be stored.')
    group.add_argument('--adaptive_seq_len',  default = False, action='store_true',
                       help='Should the sequence length be adapted to the batch during evaluation, if in fp16 the results will be slightly different due to numerical errors but greatly speed up evaluation.')
    group.add_argument('--num_fewshot', type=int, default = 0, help='Number of few-shot prompts.')
    group.add_argument('--eval_fp32',  default = False, action='store_true', help='Should the evaluation run in fp32')
    return parser

def get_model_provider():
    """Based on evaluation metric set the parallel-output flag and
    return the model provider."""

    def model_provider(pre_process=True, post_process=True):
        """Build the model."""

        print_rank_0('building GPT model ...')
        model = GPTModel(num_tokentypes=0, parallel_output=False,
                         pre_process=pre_process, post_process=post_process)

        return model

    return model_provider

def main():
    start = time.time()
    #model = load_ds_checkpoint_and_setup_megatron(extra_args_provider=tasks_args)
    model = get_model(get_model_provider(), wrap_with_ddp=False)
    args = get_args()

    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    model = model[0]

    #task_list = ALL_TASKS if args.task_list == 'all' else args.task_list.split(',')
    test_data_dir = "/nlp_data/testdb/eval_harness_data/"
    task_list = ['lambada_openai']
    #task_list = ['hellaswag']
    #task_list = ['coqa']
    #task_list = ['mnli']
    #task_list = ['wmt14-en-fr']
    #task_list = ['wikitext']
    #task_list = ['cbt-cn']
    task_dict = tasks.get_task_dict(task_list, test_data_dir)

    model.module.activation_checkpoint_interval = 0
    model._compute_loss = False
    model.fwd_outputs = []

    tokenizer = get_tokenizer()
    adaptor = EvalHarnessAdaptor(model, tokenizer)
    ### debug 
    args.num_fewshot = 0
    ### debug
    results = evaluator.evaluate(adaptor, task_dict, False, args.num_fewshot, None)

    '''if mpu.is_pipeline_last_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        print(json.dumps(results, indent=2))
        with open(args.results_path, 'w') as outfile:
            json.dump(results, outfile, indent = 4)'''
    end = time.time()
    print("result:", results)
    print("evaluation of {} ends in {:.2f} sec, or {:.2f} min, or {:.2f} hr".format(task_list, end-start, (end-start)/60.0, (end-start)/3600.0))

if __name__ == '__main__':
    main()
