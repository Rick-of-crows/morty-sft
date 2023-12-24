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

"""Pretrain GPT"""

import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron import mpu
# from megatron.data.gpt_dataset import build_train_valid_test_datasets, build_dataset_group
from megatron.data.gpt_dataset_no_concaten import build_train_valid_test_datasets, build_dataset_group
from megatron.enums import AttnMaskType
from megatron.model import GPTModel, GPTModelPipe
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids, get_prefix_indices, reweight_loss_mask_
from megatron.utils import average_losses_across_data_parallel_group

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
import subprocess

import numpy as np

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    see_memory_usage(f"Before Building Model", force=True)

    args = get_args()

    with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=args.deepspeed_config,
                             enabled=args.zero_stage == 3,
                             mpu=mpu):
        if args.deepspeed:
            model = GPTModelPipe(
                num_tokentypes=0,
                parallel_output=True,
                attn_mask_type=AttnMaskType.prefix
            )
            # This is a hack to give us a reference to get_batch_pipe from within training.py
            # We need to call model.set_batch_fn after deepspeed.initialize
            model._megatron_batch_fn = get_batch_pipe

        else:
            model = GPTModel(
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process,
                prefix_lm=True
            )
    see_memory_usage(f"After Building Model", force=True)
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Prefix
    prefix_indices = get_prefix_indices(
        tokens,
        tokenizer.eod,
        partial_prefix_indices=None,
        reset_attention_mask=args.reset_attention_mask
    )

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
        prefix_indices=prefix_indices,
        loss_on_targets_only=args.loss_on_targets_only
    )

    # weight loss_mask
    if args.reweight_loss_based_on_position_frequency:
        reweight_loss_mask_(loss_mask, tokens)

    return tokens, labels, loss_mask, attention_mask, position_ids

def get_batch_pipe(data):
    """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    
    # padding. can not pad here.
    # for i in range(len(tokens)):
    #     if len(tokens[i]) < args.seq_length:
    #         pad_tensor=torch.ones(args.seq_length-tokens[i].size()[0])*tokenizer.pad
    #         tokens[i]=torch.cat([tokens[i],pad_tensor])
    #     else:
    #         tokens[i]=tokens[i][:args.seq_length]
    #     print(len(tokens[i]))
    #     print(tokens[i])    
    
    # if int((tokens[0]==182924).sum()) != int((tokens[0]==2).sum()):
    #     if int((tokens[0]==182924).sum()) != int((tokens[0]==2).sum()) + 1:
    #         print('<eos> token num and "instruction" token num is unreasonable!') 

    # get prompt prefix indices
    partial_prefix_indices = []
    
    # for i in range(len(tokens)):
    #     # +1 for index start with 1
    #     partial_prefix_indice = (np.array(np.where(np.array(tokens[i].cpu())==1)) + 1).tolist()[0]
    #     # need to +1 as prefix index do.(prefix index used to mask attention, so +1 to attend itslef.)
    #     eod_indice = (tokens[i]==2).nonzero().squeeze(-1) + 1
    #     if len(eod_indice)>0 and len(partial_prefix_indice)>0:
    #         if eod_indice[0] < partial_prefix_indice[0]:
    #             # print("first <eos> before first Response , need to add a prefix index [0]")
    #             partial_prefix_indice = [1] + partial_prefix_indice
    #         # if eod_indice[-1] < partial_prefix_indice[-1], a <eos> token [seq_len] will added in def `get_prefix_indices`, so there need to do nothing
    #         if eod_indice[-1] > partial_prefix_indice[-1] and not tokens[i][-1]==2:
    #             # print("<eos> is not the last token, due to a <eos> token [seq_len] will added in def `get_prefix_indices`, need to add a prefix index [seq_len] here.")
    #             partial_prefix_indice.append(args.seq_length)
    #     elif len(eod_indice)>0:
    #         # remember final will have 2 <eod> in this case
    #         partial_prefix_indice.append(1)
    #         partial_prefix_indice.append(args.seq_length)
    #     elif len(eod_indice)==0 and len(partial_prefix_indice)==0:
    #         partial_prefix_indice.append(1)
    #     # else: len(partial_prefix_indice)>0, <eod> will added in def `get_prefix_indices` of index [seq_len]

    #     partial_prefix_indices.append(partial_prefix_indice)
    
    # match `Response:\n``. only support micro-batch=1. for bloom tokenizer.
    for i in range(len(tokens)):
        _partial_prefix_indice = []
        partial_prefix_indice = (tokens[i]==66673).nonzero().squeeze(-1)
        for j in partial_prefix_indice:
            if tokens[i][j-1]==105311:
                # use '\n' as the last token of prefix
                _partial_prefix_indice.append(j.tolist()+3)
                break
        # if there is no `### Response` in seqence, just skip this sample.
        if len(_partial_prefix_indice) == 0:
            _partial_prefix_indice.append(args.seq_length+1)
            
        partial_prefix_indices.append(_partial_prefix_indice)
    
    # unsqueeze the list 2d -> 1d for mask row only
    prefix_indices=[partial_prefix_indices[i][0] for i in range(len(partial_prefix_indices))]
    
    # skip prefix check in order to save time.      
    # Prefix
    # prefix_indices = get_prefix_indices(
    #     tokens,
    #     tokenizer.eod,
    #     args.pad_id,
    #     # partial_prefix_indices=None,
    #     partial_prefix_indices=partial_prefix_indices,
    #     reset_attention_mask=args.reset_attention_mask
    # )

    # Get the masks and position ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.pad_id,
        args.reset_position_ids,
        # no concaten training just have at most 1 prefix index and mask row only, please set `args.reset_attention_mask` False.
        args.reset_attention_mask,
        args.eod_mask_loss,
        prefix_indices=prefix_indices,
        loss_on_targets_only=args.loss_on_targets_only
    )

    # weight loss_mask
    if args.reweight_loss_based_on_position_frequency:
        reweight_loss_mask_(loss_mask, tokens)

    return (tokens, position_ids, attention_mask), (labels, loss_mask), prefix_indices

def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    train_ds, valid_ds, test_ds = None, None, None

    print_rank_0('> building train, validation, and test datasets for GPT ...')
    # Option 1 of data loading using --data-path

    if args.data_path:
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=args.data_path,
            data_impl=args.data_impl,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup))
    # Option 2 of data loading using --(train|valid|test)-weighted-split-paths
    elif args.train_weighted_split_paths:
        assigned_train_valid_test = []
        if args.train_weighted_split_paths is not None:
            train_ds = []
            assigned_train_valid_test.append("train")
        if args.valid_weighted_split_paths is not None:
            valid_ds = []
            assigned_train_valid_test.append("valid")
        if args.test_weighted_split_paths is not None:
            test_ds = []
            assigned_train_valid_test.append("test")

        for s in assigned_train_valid_test:
            data_groups = zip(eval(f"args.{s}_weighted_split_paths"),
                                eval(f"args.{s}_weighted_split_weights"),
                                eval(f"args.{s}_weighted_split_splits"),
                                eval(f"args.{s}_weighted_split_names"))
            for paths, weights, splits, name in data_groups:
                d = build_dataset_group(name, paths, weights, splits,
                                        args.data_impl,
                                        train_val_test_num_samples,
                                        args.seq_length, args.seed,
                                        (not args.mmap_warmup),
                                        train_valid_test=s)
                eval(f"{s}_ds").append(d)
    else:
        raise NotImplementedError("No dataloading argument passed")

    print_rank_0("> finished creating GPT datasets ...")
    return train_ds, valid_ds, test_ds

def command_exists(cmd):
    result = subprocess.Popen(f'type {cmd}', stdout=subprocess.PIPE, shell=True)
    return result.wait() == 0

def git_ds_info():
    from deepspeed.env_report import main as ds_report
    ds_report()

    # Write out version/git info
    git_hash_cmd = "git rev-parse --short HEAD"
    git_branch_cmd = "git rev-parse --abbrev-ref HEAD"
    if command_exists('git'):
        try:
            result = subprocess.check_output(git_hash_cmd, shell=True)
            git_hash = result.decode('utf-8').strip()
            result = subprocess.check_output(git_branch_cmd, shell=True)
            git_branch = result.decode('utf-8').strip()
        except subprocess.CalledProcessError:
            git_hash = "unknown"
            git_branch = "unknown"
    else:
        git_hash = "unknown"
        git_branch = "unknown"
    print(f'**** Git info for Megatron: git_hash={git_hash} git_branch={git_branch} ****')


if __name__ == "__main__":
    git_ds_info()
    pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
