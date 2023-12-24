# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPT-2 model."""

from functools import partial
import torch

from megatron import get_args
from megatron import mpu
from megatron.enums import AttnMaskType
from .module import MegatronModule, fp32_to_float16

from .language_model import parallel_lm_logits
from .language_model import get_language_model
from .utils import init_method_normal
from .utils import scaled_init_method_normal

from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec
# from megatron.model.fused_layer_norm import MixedFusedLayerNorm as LayerNorm
from megatron.model.module import float16_to_fp32
from .language_model import EmbeddingPipe
# from .transformer import ParallelTransformerLayerPipe
from .transformer_falcon import ParallelTransformerLayerPipe
from .transformer_falcon import SyncLayerNorm




def get_cross_entropy(is_prefix: bool):
    def CrossEntropy(output, labels):
        labels, loss_mask = labels[0], labels[1]

        args = get_args()

        # print(f"### device: {output.device}, output: {output.size()}, device: {labels.device}, lebels: {labels.size()}")
        losses = mpu.vocab_parallel_cross_entropy(output.contiguous().float(), labels)

        if is_prefix:
            micro_batch_size, sequence_length = loss_mask.shape
            average_tokens_per_sample: torch.Tensor
            if args.loss_on_targets_only:
                # HACK: This is useful when we obtain loss masks that are microbatch dependent. Consequently, if we want to
                #   preserve the notion that all tokens have the same impact on the loss, we can only normalise using a
                #   microbatch independent value. It should be expected weight over a microbatch.
                #   Here we still use `sequence_length`, that's batch size dependent, in order to be backwards compatible with
                #   current experiment on vanilla gpt.
                if args.reweight_loss_based_on_position_frequency:
                    reweight = torch.arange(
                        sequence_length, 0, -1, dtype=torch.float, device=loss_mask.device
                    ) / (sequence_length + 1) * 2
                    average_tokens_per_sample = reweight.flip(-1).cumsum(-1).mean()
                else:
                    average_tokens_per_sample = (sequence_length + 1) / 2
            else:
                average_tokens_per_sample = sequence_length
            expected_number_of_tokens = average_tokens_per_sample * micro_batch_size
            # add by wsf: fix prefix loss calculate, ignore the processing above
            expected_number_of_tokens = loss_mask.sum()
        else:
            expected_number_of_tokens = loss_mask.sum()

        loss_mask = loss_mask.view(-1)
        loss = torch.sum(losses.view(-1) * loss_mask) / expected_number_of_tokens
        return loss
    return CrossEntropy


class GPTModelPipe(PipelineModule,MegatronModule):
    """GPT-2 Language model."""

    def __init__(
        self,
        num_tokentypes=0,
        parallel_output=True,
        attn_mask_type: AttnMaskType = AttnMaskType.causal
    ):
        args = get_args()
        self.parallel_output = parallel_output

        init_method = init_method_normal(args.init_method_std)

        self.specs = []

        def _to_float16(inputs):
            if args.fp16:
                return fp32_to_float16(inputs, lambda v: v.half())
            elif args.bf16:
                return fp32_to_float16(inputs, lambda v: v.bfloat16())
            else:
                return inputs

        self.specs.append(_to_float16)

        # Embedding layer
        self.specs.append(TiedLayerSpec('embed',
                                        EmbeddingPipe,
                                        args.hidden_size,
                                        args.padded_vocab_size,
                                        args.hidden_dropout,
                                        init_method=init_method,
                                        num_tokentypes=num_tokentypes,
                                        tied_weight_attr='word_embeddings_weight'))

        if args.fp32_residual_connection:
            if getattr(args, 'pretrain_causal_attention', False):
                self.specs.append(lambda x: x.transpose(0, 1).contiguous().float())
            else:
                # EmbeddingPipe returns attention mask as well
                self.specs.append(lambda x: (x[0].transpose(0, 1).contiguous().float(), *x[1:]))
        else:
            if getattr(args, 'pretrain_causal_attention', False):
                self.specs.append(lambda x: x.transpose(0, 1).contiguous())
            else:
                # EmbeddingPipe returns attention mask as well
                self.specs.append(lambda x: (x[0].transpose(0, 1).contiguous(), *x[1:]))

        for layer_idx in range(args.num_layers):
            self.specs.append(
                LayerSpec(ParallelTransformerLayerPipe,
                    init_method=init_method,
                    output_layer_init_method=scaled_init_method_normal(args.init_method_std,
                                                                       args.num_layers),
                    layer_number=layer_idx,
                    # TODO: Change naming of class from GPT to something that encapsulate prefix lm.
                    self_attn_mask_type=attn_mask_type))

        # Undo data format change
        def undo(x):
            if not getattr(args, 'pretrain_causal_attention', False):
                x = x[0]
            return x.transpose(0, 1).contiguous()
        self.specs.append(undo)

        # Final layernorm after transformer layers
        self.specs.append(
            LayerSpec(SyncLayerNorm,
                      args.hidden_size,
                      eps=args.layernorm_epsilon))

        def _logits_helper(embedding, lm_output):
            """A wrapper to massage inputs/outputs from pipeline. """
            return parallel_lm_logits(
                lm_output,
                embedding.word_embeddings_weight,
                self.parallel_output)

        self.specs.append(
            TiedLayerSpec('embed',
                          EmbeddingPipe,
                          args.hidden_size,
                          args.padded_vocab_size,
                          args.hidden_dropout,
                          init_method=init_method,
                          num_tokentypes=num_tokentypes,
                          forward_fn=_logits_helper,
                          tied_weight_attr='word_embeddings_weight')
        )

        # Convert to fp32 if needed
        if args.fp16 or args.bf16:
            self.specs.append(float16_to_fp32)

        if args.checkpoint_activations:
            interval = args.checkpoint_num_layers
        else:
            interval = 0

        from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
        topo = PipeModelDataParallelTopology(num_pp=mpu.get_pipeline_model_parallel_world_size(),
                                             num_mp=mpu.get_tensor_model_parallel_world_size(),
                                             num_dp=mpu.get_data_parallel_world_size())

        # here one can extend the regex to include more layers to be counted towards partitioning,
        # e.g. 'type:transformer|embedding' will add up all the transformer blocks and also the first
        # and last embedding layers and then partition that transformers+2 layers - so to get a good
        # balance you may want to use less transformer layers
        #
        # caveat emptor: the current implementation of PP fails unless each stage has at least one
        # transformer layer
        if args.pp_partition_method is not None:
            partition_method = args.pp_partition_method
        else:
            partition_method = 'type:transformer'

        super().__init__(layers=self.specs,
                         loss_fn=get_cross_entropy(is_prefix=attn_mask_type is AttnMaskType.prefix),
                         topology=topo,
                         activation_checkpoint_interval=interval,
                         partition_method=partition_method)
