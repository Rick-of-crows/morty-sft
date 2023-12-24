# under the license https://huggingface.co/spaces/bigscience/license

from functools import reduce
from logging import logMultiprocessing
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir,os.path.pardir,os.path.pardir)))

from lm_eval.models.gpt2 import GPT2LM
from lm_eval import evaluator, tasks, utils
from lm_eval.base import CacheHook
from tqdm import tqdm
import torch.nn.functional as F

from lm_eval.tasks import ALL_TASKS
#from pretrain_gpt import model_provider
import numpy as np
import time
import torch
from sentencepiece import SentencePieceProcessor

from nemo.core.config import hydra_runner
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel, initialize_model_parallel_for_nemo
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.utils import logging
from nemo.utils.app_state import AppState
from nemo.utils.model_utils import inject_model_parallel_rank
from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.modules.common.text_generation_strategy import model_inference_strategy_dispatcher
from nemo.collections.nlp.modules.common.text_generation_utils import megatron_gpt_generate, synced_generate
#from nemo.collections.nlp.modules.common.lm_utils import pad_batch
from nemo.utils.distributed import initialize_distributed

from apex.transformer.pipeline_parallel.schedules.common import build_model
from apex.transformer import parallel_state, tensor_parallel
from nemo.utils.exp_manager import exp_manager

import pickle
import json
from typing import List

class EvalHarnessAdaptor(GPT2LM):
    def __init__(self, model, tokenizer, cfg):
        self.model = model
        self.is_main = self.model.local_rank == 0
        self._device = torch.device(f"cuda:{self.model.local_rank}")
        print("local_rank:", self.model.local_rank)
        self.model.tokenizer = tokenizer
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.VOCAB_SIZE = self.tokenizer.n_words
        self.EOT_TOKEN_ID = self.tokenizer.eos_id
        self.model = Float16Module(module=self.model, precision=cfg.trainer.precision)

        self._max_length = self.cfg.model.encoder_seq_length
        # For ds we split into mini batches and then micro batches to keep pipelining api happy.
        # With Megatron we just go to micro_batches directly
        self._batch_size = self.cfg.model.micro_batch_size

        self.cache_hook = CacheHook(None)
        
        self.is_model_parallel = parallel_state.get_tensor_model_parallel_world_size() > 1
        self.is_pipe_parallel = parallel_state.get_pipeline_model_parallel_world_size() > 1
        #self.dp_world_size = parallel_state.get_data_parallel_world_size()
        #self.dp_rank = parallel_state.get_data_parallel_rank()

        self.is_last_stage = True if not self.is_pipe_parallel else parallel_state.is_pipeline_last_stage()  # only the last stage of the pipeline model will receive the logits

        self.generate_max_len = self.cfg.model.data.max_generate_tokens

        self.inference_strategy = model_inference_strategy_dispatcher(self.model.module)


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
                    token_list=self.tok_encode(string),
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

                    # since in _collate we make sure length is descending, the longest is always the first one.
                    padding_length = (
                        padding_length if padding_length is not None else inplen
                    )

                    # pad to length
                    #print("padding_length:", padding_length, " input:", inplen)
                    inp = torch.cat(
                        [
                            inp,  # [seq]
                            torch.zeros(padding_length - inplen, dtype=torch.long).to(
                                inp.device
                            ),  # [padding_length - seq]
                        ],
                        dim=0,
                    )
                    '''if inplen < self._max_length:
                      num_pad = (self._max_length - inplen)
                      pad_tensor = torch.tensor([self.EOT_TOKEN_ID] * num_pad, dtype=torch.long).to(self.device)
                      inp = torch.cat([inp, pad_tensor], dim=0)'''
                    inps.append(inp.unsqueeze(0))
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

        return reord.get_original(res)

    def greedy_until(self, requests):
        """
        Greedy until is lm_eval harness' way to say "do greedy generation" - necessary for some tasks.
        the eval harness dispatches requests to the model, and the model does argmax generation, the results of which
        are returned to the eval harness to evaluate.
        TODO: batched / data parallel generation
        :param requests: Dictionary of requests containing the context (prompt) and 'until' - a token or
                         list of stop tokens.
        """
        res = []
        def _collate(x):
            toks = self.tok_encode(x[0], True, False)
            return (-len(toks), x[0])

        reord = utils.Reorderer(requests, _collate)

        for chunk in utils.chunks(
            tqdm(reord.get_reordered()), self.batch_size
        ):
            #print("chunk:", len(chunk))
            context_tokens, inplens, inps, untils = [], [], [], []
            for context, until in chunk:
                #print("context:", context, " until:", len(until))
                #context = "I believe the meaning of life is simply put, the theory of relativity states that Building a website can be done"
                enc = self.tok_encode(context, True, False)
                #print("enc:", len(enc))
                if len(enc) + self.generate_max_len > self._max_length:
                    enc = enc[-(self._max_length - self.generate_max_len):]
                    #context = self.tok_decode(new_enc)
                inps.append(context)
                untils.append(until)
                context_tokens.append(enc)
                inplens.append(len(enc))

            #response = megatron_gpt_generate(self.model.module, inps, self.tokenizer, self.length_params, self.sampling_params)
            context_tokens, context_lengths = pad_batch(context_tokens, self.tokenizer.eos_id, self.generate_max_len)
            context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
            context_length_tensor = torch.cuda.LongTensor(context_lengths)
            #print("context:", context_tokens_tensor.shape)
            output = synced_generate(self.model.module, self.inference_strategy, context_tokens_tensor, \
                context_length_tensor, self.generate_max_len, all_probs=False, temperature=0.0, \
                top_k=0, top_p=0.0, greedy=True, repetition_penalty=1.0, min_tokens_to_generate=1, \
            )
            decode_tokens, output_logits, full_logits = output
            decode_tokens = decode_tokens.cpu().numpy().tolist()
            for idx in range(len(decode_tokens)):
                decode_token = decode_tokens[idx]
                #print("decode:",self.tok_decode(decode_token).strip())
                s = self.tok_decode(decode_token[inplens[idx]:]).strip()
                for term in untils[idx]:
                    s = s.split(term)[0]
                self.cache_hook.add_partial("greedy_until", (inps[idx], untils[idx]), s)
                res.append(s)

        return reord.get_original(res)

    def create_model_inputs(self, tokens):

        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            self.EOT_TOKEN_ID,
            self.cfg.model.data.get('reset_position_ids', False),
            self.cfg.model.data.get('reset_attention_mask', False),
            self.cfg.model.data.get('eod_mask_loss', False))

        return (tokens, position_ids, attention_mask), (tokens, loss_mask)

    def _model_call(self, inps):
        # Since the shape of the micro-batch will change
        # We need set the correct shapes here
        # So that latter pipeline stages knows which shapes to expect.
        # Otherwise we will deadlock.

        # scatter inputs to all dp ranks:
        #inps, padded = self._dp_scatter(inps)

        ## debug

        output = self.model(*self.create_model_inputs(inps)[0], labels=None)
        if self.is_model_parallel:
            output = tensor_parallel.gather_from_tensor_model_parallel_region(output)[..., :self.VOCAB_SIZE]
        # gather outputs from all dp ranks:
        #output = self._dp_gather(output)

        return output

def pad_batch(batch, pad_id, max_len):
    context_lengths = []
    max_context_length = max([len(tokens) for tokens in batch])
    for tokens in batch:
        context_length = len(tokens)
        if context_length < max_context_length + max_len:
            tokens.extend([pad_id] * (max_context_length + max_len - context_length))
        context_lengths.append(context_length)
    return batch, context_lengths

def override_args(args, override_args, skip_keys, skip_if_specified_keys):
    for k, v in vars(override_args).items():
        if k in skip_keys:
            continue
        if k in skip_if_specified_keys and getattr(args, k) is not None:
            continue
        setattr(args, k, v)

class LLaMaTokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.vocab_size = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

@hydra_runner(config_path="conf", config_name="megatron_gpt_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')
    start = time.time()
    # trainer required for restoring model parallel models
    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    initialize_distributed(trainer)
    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
        == cfg.model.tensor_model_parallel_size * cfg.model.pipeline_model_parallel_size
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"
    

    task_list = cfg.model.data.task_list
    assert(len(task_list) > 0), "Not support empty task_list."
    #test_data_dir = cfg.model.data.test_data_dir
    task_dict = tasks.get_task_dict(task_list)
    print("do task_list:", task_list)
    
    ## cfg.model.data.gpt_model_file
    if cfg.model.data.gpt_model_file:
        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.model.data.gpt_model_file):
            save_restore_connector.model_extracted_dir = cfg.model.data.gpt_model_file

        pretrained_cfg = MegatronGPTModel.restore_from(
            restore_path=cfg.model.data.gpt_model_file,
            trainer=trainer,
            return_config=True,
            save_restore_connector=save_restore_connector,
        )
        #print("pretrain_cfg:", pretrained_cfg)
        #pretrained_cfg = set_tokenizer(pretrained_cfg, cfg)
        #print("afterjon:", pretrained_cfg)
        OmegaConf.set_struct(pretrained_cfg, True)
        with open_dict(pretrained_cfg):
            pretrained_cfg.sequence_parallel = False
            pretrained_cfg.activations_checkpoint_granularity = None
            pretrained_cfg.activations_checkpoint_method = None
        model = MegatronGPTModel.restore_from(
            restore_path=cfg.model.data.gpt_model_file,
            trainer=trainer,
            override_config_path=pretrained_cfg,
            save_restore_connector=save_restore_connector,
        )        
    elif cfg.model.data.checkpoint_dir:
        app_state = AppState()
        if cfg.model.tensor_model_parallel_size > 1 or cfg.model.pipeline_model_parallel_size > 1:
            app_state.model_parallel_size = cfg.model.tensor_model_parallel_size * cfg.model.pipeline_model_parallel_size
            app_state.tensor_model_parallel_size = cfg.model.tensor_model_parallel_size
            app_state.pipeline_model_parallel_size = cfg.model.pipeline_model_parallel_size


        initialize_model_parallel_for_nemo(world_size=trainer.world_size, \
                global_rank=trainer.global_rank, \
                local_rank=trainer.local_rank, \
                tensor_model_parallel_size=cfg.model.tensor_model_parallel_size, \
                pipeline_model_parallel_size=cfg.model.pipeline_model_parallel_size, \
                virtual_pipeline_model_parallel_size=cfg.model.virtual_pipeline_model_parallel_size, \
                pipeline_model_parallel_split_rank=0, \
                micro_batch_size=cfg.model.micro_batch_size, \
                global_batch_size=cfg.model.global_batch_size, \
                seed=cfg.model.seed, \
                apex_transformer_log_level=cfg.model.apex_transformer_log_level, 
            )

        checkpoint_path = inject_model_parallel_rank(os.path.join(cfg.model.data.checkpoint_dir, cfg.model.data.checkpoint_name))
        model = MegatronGPTModel.load_from_checkpoint(checkpoint_path, hparams_file=cfg.model.hparams_file, trainer=trainer)

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size_=cfg.model.tensor_model_parallel_size,
        pipeline_model_parallel_size_=cfg.model.pipeline_model_parallel_size,
    )
    
    print("load done...")

    tokenizer = LLaMaTokenizer(model_path="/nlp_data/zoo/pretrain/llama_model/tokenizer.model")

    adaptor = EvalHarnessAdaptor(model, tokenizer, cfg)
    results = evaluator.evaluate(adaptor, task_dict, False, cfg.model.data.num_fewshot, None)

    '''if mpu.is_pipeline_last_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        print(json.dumps(results, indent=2))
        with open(args.results_path, 'w') as outfile:
            json.dump(results, outfile, indent = 4)'''
    end = time.time()
    print("result:", results)
    print("evaluation of {} ends in {:.2f} sec, or {:.2f} min, or {:.2f} hr".format(task_list, end-start, (end-start)/60.0, (end-start)/3600.0))

if __name__ == '__main__':
    main()
