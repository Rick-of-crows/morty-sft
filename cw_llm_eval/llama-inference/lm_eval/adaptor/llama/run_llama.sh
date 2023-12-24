set -x

NUM_NODES=1

MASTER_ADDR=localhost
MASTER_PORT=6000

TEST_DATA_DIR="/nlp_data/testdb/eval_harness_data/"

export SACREBLEU=${TEST_DATA_DIR}
export HF_DATASETS_CACHE=${TEST_DATA_DIR}
export CUDA_VISIBLE_DEVICES=6

CKPT_DIR="/nlp_data/zoo/pretrain/llama_model/7B/"
TOKENIZER_PATH="/nlp_data/zoo/pretrain/llama_model/tokenizer.model"


NUM_FEWSHOT=0
MAX_GENERATE_NUM=32
SEQ_LEN=1024
BATCH_SIZE=1

#TASK_LIST="lambada_openai"
#TASK_LIST="cbt-ne cbt-cn hellaswag"
#TASK_LIST="lambada_openai"
## fewshot
#TASK_LIST="piqa arc_easy arc_challenge race_middle race_high coqa rte wsc273 winogrande triviaqa"
#TASK_LIST="wsc273 winogrande"
#TASK_LIST="wmt14-en-fr"
#TASK_LIST="cbt-ne cbt-cn hellaswag wikitext wmt14-en-fr wmt14-fr-en cnn_dailymail"
#TASK_LIST="race_middle race_high arc_easy arc_challenge"
#TASK_LIST="rte"
#TASK_LIST="coqa"
#TASK_LIST="truthfulqa_mc"
TASK_LIST="c3 cluewsc wiki_zh"


torchrun --nproc_per_node ${NUM_NODES} \
     --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} \
     ./adaptor/llama/llama_eval.py \
     -ckpt_dir ${CKPT_DIR} \
     -tokenizer_path ${TOKENIZER_PATH} \
     -task_list ${TASK_LIST} \
     -num_fewshot ${NUM_FEWSHOT} \
     -max_seq_len ${SEQ_LEN} \
     -max_generate_tokens ${MAX_GENERATE_NUM} \
     -max_batch_size ${BATCH_SIZE} \
