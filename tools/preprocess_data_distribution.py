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

"""Processing data for pretraining."""

import argparse
import json
import multiprocessing
import os
import sys
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from megatron.data.indexed_dataset import best_fitting_dtype
import time

import torch
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset


# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

cnt = 0

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            splitter = nltk.load("tokenizers/punkt/english.pickle")
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text = splitter._params,
                    lang_vars = CustomLanguageVars())
            else:
                Encoder.splitter = splitter

        else:
            Encoder.splitter = IdentitySplitter()

    def encode(self, json_line):
        data = json.loads(json_line)
        ids = {}
        global cnt
        for key in self.args.json_keys:
            text = data[key]
            doc_ids = []
            for sentence in Encoder.splitter.tokenize(text):
                sentence_ids = Encoder.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.append(sentence_ids)
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids[-1].append(Encoder.tokenizer.eod)
                cnt += 1
            else:
                print(doc_ids)
            ids[key] = doc_ids
        # print('{} <eod> tokens are added'.format(cnt))
        # if cnt == 52003:
        #     print("cnt == 52003")
        return ids, len(json_line)

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str,
                       help='Path to input JSON')
    group.add_argument('--output', type=str,
                       help='Path to output JSON')
    group.add_argument('--datasets', nargs='+', default=None,
                       help='Paths to one or more input datasets to merge')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=True,
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer', 'PretrainedFromHF', 'LlamaTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument("--tokenizer-name-or-path", type=str, default=None,
                       help="Name or path of the huggingface tokenizer.")
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                            'This is added for computational efficieny reasons.')
    group.add_argument('--pad-vocab-size-to', type=int, default=None,
                       help='Pad the vocab size to be divisible by this value.'
                            'Value of the size of the vocabulary of the tokenizer to reach. This value must be greater than'
                            ' the initial size of the tokenizer. If this argument is used the value of '
                            '`make-vocab-size-divisible-by` will be ignored.')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, # required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes to launch')
    group.add_argument('--log-interval', type=int, default=100,
                       help='Interval between progress updates')
    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert'):
        if not args.split_sentences:
            print("Bert tokenizer detected, are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args

def get_distribution(args, tokenizer):
    cnt_1, cnt_2, cnt_3, cnt_4, cnt_5=0, 0, 0, 0, 0 
    with open(args.input, 'r', encoding='utf-8') as fr:
        """统计训练数据长度"""
        # for line in fr:
        #     doc = json.loads(line)
            # for _, value in doc.items():
            #     wordscount=len(tokenizer.tokenize(value))
            #     if wordscount < 512:
            #         cnt_1+=1
            #     elif wordscount < 1024:
            #         cnt_2+=1
            #     elif wordscount < 2048:
            #         cnt_3+=1
            #     else:
            #         cnt_4+=1
        #     print(f"Data distribution: {cnt_1} seqs < 512; {cnt_2} seqs < 1024; {cnt_3} seqs < 2048; {cnt_4} seqs > 2048")
            
        """统计原始数据答案长度"""
        # doc = json.load(fr)
        # for d in doc:

        for line in fr:
            d = json.loads(line)
            v = d["output"]
            if "图" in d["output"] or "所示" in d["output"] or "展示" in d["output"] or "模型" in d["output"] \
                or "图" in d["instruction"] or "所示" in d["instruction"] \
                    or "展示" in d["instruction"] or "模型" in d["instruction"]:
                continue
            wordscount=len(tokenizer.tokenize(v))
            if wordscount < 20:
                cnt_1+=1
            elif wordscount < 50:
                cnt_2+=1
            elif wordscount < 100:
                cnt_3+=1
            elif wordscount < 150:
                cnt_4+=1
            else:
                cnt_5+=1          
        print(f"Data distribution: {cnt_1} seqs < 20; {cnt_2} seqs < 50; {cnt_3} seqs < 100; {cnt_4} seqs < 150; {cnt_5} seqs > 150")           

import random

def sample_with_distribution(args, tokenizer):
    cnt1,cnt2,cnt3,cnt4 = 0,0,0,0
    with open(args.input, 'r', encoding='utf-8') as fr:
        doc = json.load(fr)
        new_doc=[]
        for d in doc:

        # new_doc = []  
        # for line in fr:
        #     d = json.loads(line)

            # v = d["output"]
            # wordscount=len(tokenizer.tokenize(v))
            # if wordscount > 100:
            #     new_doc.append(d)
            # else:
            #     if(random.random()) > 0.0244:
            #         continue
            #     else:
            #         new_doc.append(d)
            #         cnt1+=1

            v = d["output"]
            wordscount=len(tokenizer.tokenize(v))
            if wordscount > 50:
                if(random.random()) > 0.8523:
                    continue
                else:
                    new_doc.append(d)
                    cnt1+=1

            # if wordscount > 100:
            #     if(random.random()) > 0.5:
            #         continue
            #     else:
            #         new_doc.append(d)
            #         cnt1+=1
            # elif wordscount > 50:
            #     if(random.random()) > 0.3743:
            #         continue
            #     else:
            #         new_doc.append(d)
            #         cnt2+=1
            # elif wordscount > 50:
            #     if(random.random()) > 0.1713:
            #         continue
            #     else:
            #         new_doc.append(d)
            #         cnt3+=1
            # else:
            #     if(random.random()) > 0.2156:
            #         continue
            #     else:
            #         new_doc.append(d)
            #         cnt4+=1
    
    print(cnt1,cnt2,cnt3,cnt4)
    with open(args.output, 'w', encoding='utf-8') as fw:
        json.dump(new_doc, fw, ensure_ascii=False, indent=4)
        # for nd in new_doc:
        #     json.dump(nd, fw, ensure_ascii=False)
        #     fw.write('\n')

    


def main():
    args = get_args()
    startup_start = time.time()

    # print("Opening", args.input)
    # fin = open(args.input, 'r', encoding='utf-8')

    if nltk_available and args.split_sentences:
        nltk.download("punkt", quiet=True)
        
    hf_tokenizer_kwargs = {}
    if args.vocab_extra_ids > 0:
        # TODO @thomasw21 we might need to concatenate to a pre-existing list?
        hf_tokenizer_kwargs["additional_special_tokens"] = [f"<extra_id_{_id}>" for _id in range(args.vocab_extra_ids)]
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, **hf_tokenizer_kwargs)
    
    get_distribution(args, tokenizer)

    # sample_with_distribution(args, tokenizer)

    
    return 


    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    encoded_docs = pool.imap(encoder.encode, fin, 25)
    #encoded_docs = map(encoder.encode, fin)

    level = "document"
    if args.split_sentences:
        level = "sentence"

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix,
                                                    key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,
                                                    key, level)
        builders[key] = indexed_dataset.make_builder(output_bin_files[key],
                                                     impl=args.dataset_impl,
                                                     dtype=best_fitting_dtype(tokenizer.vocab_size))

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)
    
    print("args.json-keys: {}".format(args.json_keys))

    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed
        for key, sentences in doc.items():
            if len(sentences) == 0:
                continue
            for sentence in sentences:
                builders[key].add_item(torch.IntTensor(sentence))
            builders[key].end_document()
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {i} documents",
                f"({i/elapsed} docs/s, {mbs} MB/s).",
                file=sys.stderr)

    for key in args.json_keys:
        builders[key].finalize(output_idx_files[key])

if __name__ == '__main__':
    main()
