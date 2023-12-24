set -x

TEST_DATA_DIR="/nlp_data/testdb/eval_harness_data/"

export SACREBLEU=${TEST_DATA_DIR}
export HF_DATASETS_CACHE=${TEST_DATA_DIR}
export CUDA_VISIBLE_DEVICES=7

#MODEL_DIR="/nlp_data/zoo/pretrain/huggingface/bloom/bloom-560m/"
#MODEL_DIR="/nlp_data/zoo/pretrain/huggingface/bloom/bloomz-7b1/"
MODEL_DIR="/data_ssd/step336700_token397B/"
#MODEL_DIR="/nlp_data/zoo/pretrain/huggingface/gpt2-medium/"
#MODEL_DIR="/nlp_data/zoo/pretrain/huggingface/falcon/7B/"
#MODEL_DIR="/home/yckj0239/bloom-560m/"

NUM_FEWSHOT=0
MAX_GENERATE_NUM=32
SEQ_LEN=2048
BATCH_SIZE=8
DEVICE_ID=0
RATIO=1.0
# proportion

#TASK_LIST="lambada_openai wikitext cbt-cn cbt-ne wsc273 coqa"
#TASK_LIST="truthfulqa_mc_zh"
#TASK_LIST="cbt-ne cbt-cn hellaswag"
#TASK_LIST="lambada_openai cbt-ne cbt-cn hellaswag wikitext piqa arc_easy arc_challenge race_middle race_high coqa rte wsc273 winogrande triviaqa wmt14-en-fr wmt14-fr-en cnn_dailymail"
#TASK_LIST="wikitext piqa arc_easy arc_challenge race_middle race_high"
#TASK_LIST="coqa"
#TASK_LIST="wiki_zh"
#TASK_LIST="race_middle race_high"
#TASK_LIST="c3 cluewsc wiki_zh"
#TASK_LIST="hellaswag"
#TASK_LIST="arc_challenge"
#TASK_LIST="truthfulqa_mc"
#TASK_LIST="hendrycksTest-*"
#TASK_LIST="AGIEval-*"
TASK_LIST="ceval-*"

## fewshot
#TASK_LIST="piqa arc_easy arc_challenge race_middle race_high coqa rte wsc273 winogrande triviaqa"
#TASK_LIST="wsc273 winogrande"
#TASK_LIST="wmt14-en-fr"
#TASK_LIST="cbt-ne cbt-cn hellaswag wikitext wmt14-en-fr wmt14-fr-en cnn_dailymail"
#TASK_LIST="race_middle race_high arc_easy arc_challenge"
#TASK_LIST="rte"
#TASK_LIST="coqa"

python ./adaptor/bloom/bloom_eval.py \
     -model_dir ${MODEL_DIR} \
     -task_list ${TASK_LIST} \
     -num_fewshot ${NUM_FEWSHOT} \
     -max_seq_len ${SEQ_LEN} \
     -device_id ${DEVICE_ID} \
     -max_generate_tokens ${MAX_GENERATE_NUM} \
     -max_batch_size ${BATCH_SIZE} \
     -ratio ${RATIO} \
