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

"""Dataloaders."""

import torch

from megatron import get_args
from megatron import mpu
from megatron.data.decoder_packed_mtf_dataset import DecoderPackedMTFDataset


def build_pretraining_data_loader(dataset, consumed_samples, num_workers=None):
    """Buld dataloader given an input dataset."""

    if dataset is None:
        return None
    args = get_args()

    # Megatron sampler
    if args.dataloader_type == 'single':
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size())
    elif args.dataloader_type == 'cyclic':
        batch_sampler = MegatronPretrainingRandomSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size())
    else:
        raise Exception('{} dataloader type is not supported.'.format(
            args.dataloader_type))

    if num_workers is None:
        num_workers = args.num_workers

    # Torch dataloader.
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=default_collate if args.use_padding else None,
        pin_memory=True
    )

class MegatronPretrainingSampler:

    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]


class MegatronPretrainingRandomSampler:

    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.last_batch_size = \
            self.total_samples % self.micro_batch_times_data_parallel_size

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        # data sharding and random sampling
        bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) \
                      * self.micro_batch_size
        bucket_offset = current_epoch_samples // self.data_parallel_size
        start_idx = self.data_parallel_rank * bucket_size

        g = torch.Generator()
        g.manual_seed(self.epoch)
        random_idx = torch.randperm(bucket_size, generator=g).tolist()
        idx_range = [start_idx + x for x in random_idx[bucket_offset:]]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []

import re
if torch.__version__ == "1.8.1+cu101":
    from torch._six import container_abcs, string_classes, int_classes
else:
    # for torch 1.13
    import collections.abc as container_abcs
    string_classes=str
    int_classes=int

np_str_obj_array_pattern = re.compile(r'[SaUO]')

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    args = get_args()
    
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        # padding.
        max_len = 0
        if args.use_dynamic_padding:
            for i in range(len(batch)):
                max_len = max(batch[i].size()[0], max_len)
            max_len = min(max_len, args.seq_length)
        else: 
            max_len = args.seq_length
            
        for i in range(len(batch)):
            # if batch[i].size()[0] > args.seq_length:
            #     batch[i] = batch[i][:args.seq_length]
            if batch[i].size()[0] >= max_len:
                batch[i] = batch[i][:max_len]
                # print("### trunt the tensor")
            else:
                # batch[i] = torch.cat([batch[i], torch.tensor((args.seq_length-batch[i].size()[0])*[args.pad_id], dtype=elem.dtype)])
                batch[i] = torch.cat([batch[i], torch.tensor((max_len-batch[i].size()[0])*[args.pad_id], dtype=elem.dtype)])
                # print("### pad the tensor, tensor shape: ", batch[i].size())
        
        elem = batch[0]
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            # print("### coming in torch.utils.data.get_worker_info()")
            numel = sum([x.numel() for x in batch])

            if torch.__version__ == "1.8.1+cu101":
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            else:
                # compatible for torch version >= 1.13
                storage = elem.storage()._new_shared(numel, device=elem.device)
                out = elem.new(storage).resize_(len(batch), *list(elem.size()))

        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))