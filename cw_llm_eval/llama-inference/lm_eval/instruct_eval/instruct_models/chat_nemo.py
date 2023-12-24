import torch
from .base_model import BaseM
import os

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
from nemo.utils.get_rank import is_global_rank_zero
#from nemo.collections.nlp.modules.common.lm_utils import pad_batch
from nemo.utils.distributed import initialize_distributed

from apex.transformer.pipeline_parallel.schedules.common import build_model
from apex.transformer import parallel_state, tensor_parallel
from nemo.utils.exp_manager import exp_manager

class ChatNeMo(BaseM):
    def __init__(self, config):
        self.config = config
        self.config_path = config['config_path']
        self.init_generate_template()
        self.init_model_from_config()

    def init_sampling_params(self, cfg):
        self.length_params: LengthParam = {
            "max_length": cfg.inference.tokens_to_generate,
            "min_length": cfg.inference.min_tokens_to_generate,
        }

        self.sampling_params: SamplingParam = {
            "use_greedy": cfg.inference.greedy,
            "temperature": cfg.inference.temperature,
            "top_k": cfg.inference.top_k,
            "top_p": cfg.inference.top_p,
            "repetition_penalty": cfg.inference.repetition_penalty,
            "add_BOS": cfg.inference.add_BOS,
            "all_probs": cfg.inference.all_probs,
            "compute_logprob": cfg.inference.compute_logprob,
            "end_strings": cfg.inference.end_strings
        }

    def init_model_from_config(self):
        cfg = OmegaConf.load(self.config_path)
        logging.info("\n\n************** Experiment configuration ***********")
        logging.info(f'\n{OmegaConf.to_yaml(cfg)}')
        trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)
        ### init sampling param
        self.init_sampling_params(cfg)
        print("init sampling params done...")
        if cfg.gpt_model_file:
            save_restore_connector = NLPSaveRestoreConnector()
            if os.path.isdir(cfg.gpt_model_file):
                save_restore_connector.model_extracted_dir = cfg.gpt_model_file

            pretrained_cfg = MegatronGPTModel.restore_from(
                restore_path=cfg.gpt_model_file,
                trainer=trainer,
                return_config=True,
                save_restore_connector=save_restore_connector,
            )
            OmegaConf.set_struct(pretrained_cfg, True)
            with open_dict(pretrained_cfg):
                pretrained_cfg.sequence_parallel = False
                pretrained_cfg.activations_checkpoint_granularity = None
                pretrained_cfg.activations_checkpoint_method = None
            self.model = MegatronGPTModel.restore_from(
                restore_path=cfg.gpt_model_file,
                trainer=trainer,
                override_config_path=pretrained_cfg,
                save_restore_connector=save_restore_connector,
            )
        elif cfg.checkpoint_dir:
            app_state = AppState()
            if cfg.tensor_model_parallel_size > 1 or cfg.pipeline_model_parallel_size > 1:
                app_state.model_parallel_size = cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
                app_state.tensor_model_parallel_size = cfg.tensor_model_parallel_size
                app_state.pipeline_model_parallel_size = cfg.pipeline_model_parallel_size
                (
                    app_state.tensor_model_parallel_rank,
                    app_state.pipeline_model_parallel_rank,
                    app_state.model_parallel_size,
                    app_state.data_parallel_size,
                    app_state.pipeline_model_parallel_split_rank,
                    app_state.virtual_pipeline_model_parallel_rank,
                ) = fake_initialize_model_parallel(
                    world_size=app_state.model_parallel_size,
                    rank=trainer.global_rank,
                    tensor_model_parallel_size_=cfg.tensor_model_parallel_size,
                    pipeline_model_parallel_size_=cfg.pipeline_model_parallel_size,
                    pipeline_model_parallel_split_rank_=cfg.pipeline_model_parallel_split_rank,
                )
            checkpoint_path = inject_model_parallel_rank(os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name))
            self.model = MegatronGPTModel.load_from_checkpoint(checkpoint_path, hparams_file=cfg.hparams_file, trainer=trainer)
        else:
            raise ValueError("need at least a nemo file or checkpoint dir")
        print("load model done...")


    def init_generate_template(self):
        self.generate_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request. Instruction: {instruction} Response:"
        if "generate_template" in self.config and 'instruction' in self.config["generate_template"]:
            self.generate_template = self.config["generate_template"]

    def init_chain_rank(self):
        pass

    def init_chain_score(self):
        pass

    def base_generate(self, texts):
        texts = [text['instruction'] for text in texts]
        input_texts = []
        for text in texts:
            input_text = self.generate_template.format(instruction=text)
            input_texts.append(input_text)
        #print("input_texts:", input_texts)
        responses = self.model.generate(input_texts, length_params=self.length_params, sampling_params=self.sampling_params)
        #print("responses:", responses['sentences'])
        outputs = []
        if is_global_rank_zero():
            sentences = responses['sentences']
            for idx, sentence in enumerate(sentences):
                output = sentence[len(input_texts[idx]):].strip()
                outputs.append(output)
        else:
            return None
        return outputs

    def rank_generate(self, x, y1, y2):
        pass

    def score_generate(self, x, y):
        pass