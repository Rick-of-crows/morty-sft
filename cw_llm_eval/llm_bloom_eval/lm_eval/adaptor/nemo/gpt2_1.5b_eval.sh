set -x

# export NCCL_NSOCKS_PERTHREAD=4
# export NCCL_SOCKET_NTHREADS=2

NUM_NODES=1
NODE_RANK=0

GPUS_PER_NODE=1

MASTER_ADDR=localhost
MASTER_PORT=6104


### set VOCAL_FILE and MERGE_FILE in megatron_gpt_config.yaml 
#VOCAB_FILE=/nlp_data/meta/gpt2/gpt2-vocab.json
#MERGE_FILE=/nlp_data/meta/gpt2/gpt2-merges.txt

MODEL_DIR="/home/wangxi/nlp/nemo_0523/models/checkpoints/"
MODEL_NAME="last.ckpt"
# MODEL_DIR="/nlp_data/zoo/pretrain/llama.nemo/30B/tp4pp2/"
# MODEL_NAME="llama.ckpt"
#NEMO_FILE="/nlp_data/zoo/pretrain/huggingface/nemo_gpt_345m/megatron_gpt_345m.nemo"
#NEMO_FILE="/home/wangxi/models/nlp/nemo/huggingface/nemo_gpt1.3B_fp16.nemo"
#NEMO_FILE="/home/yckj0239/xx_checkpoints/step28000.nemo"
#NEMO_FILE="/nlp_data/zoo/pretrain/huggingface/nemo_gpt5b_tp2/nemo_gpt5B_fp16_tp2.nemo"
#NEMO_FILE="/home/yckj0239/40318880.nemo"

### hparam_file for loading ckpt.
HPARAM_FILE="/home/wangxi/nlp/nemo_0523/models/version_0/hparams.yaml"
#HPARAM_FILE="/home/wangxi/nlp/nemo_0523/pretraining/experiments/baseline/version_2/hparams.yaml"
#HPARAM_FILE="/home/wangxi/nlp/nemo_0523/pretraining/experiments/wx_rmt/version_99/hparams.yaml"

TEST_DATA_DIR="/nlp_data/testdb/eval_harness_data/"

export SACREBLEU=${TEST_DATA_DIR}
export HF_DATASETS_CACHE=${TEST_DATA_DIR}
export CUDA_VISIBLE_DEVICES=0

NUM_FEWSHOT=0
MAX_GENERATE_NUM=32
SEGMENTS=1
ANSIDX=0
RATIO=0.4
FORWARD_SINGLE=False

#TASK_LIST=['cbt-cn','cbt-ne','arc_easy','wsc273','arc_challenge','race_middle','race_high','wmt14-en-fr','wmt14-fr-en','coqa','hellaswag']
#TASK_LIST=['piqa','arc_easy','arc_challenge','race_middle','race_high','coqa','rte','wsc273','winogrande','triviaqa']
#TASK_LIST=['cnn_dailymail','triviaqa']
#TASK_LIST=['lambada_openai','hellaswag','piqa']
#TASK_LIST=['wsc273','winogrande']
#TASK_LIST=['hellaswag']
#TASK_LIST=['lambada_openai']
TASK_LIST=['race_middle_long_ctx']
#TASK_LIST=['mnli']
#TASK_LIST=['boolq']
#TASK_LIST=['coqa']
#TASK_LIST=['coqa','wmt14-en-fr','wmt14-fr-en']
#TASK_LIST=['squad2']
#TASK_LIST=['arc_easy','arc_challenge','openbookqa']
#TASK_LIST=['cnn_dailymail']
#TASK_LIST=['cbt-cn','cbt-ne','hellaswag']
#TASK_LIST=['wikitext']
#TASK_LIST=['triviaqa']
#TASK_LIST=['truthfulqa_mc']
#TASK_LIST=['race_middle','race_high']
#TASK_LIST=['cmnli']
#TASK_LIST=['wiki_zh']
#TASK_LIST=['c3','cluewsc','wiki_zh']
# TASK_LIST=["cmmlu-*"]
# TASK_LIST=["hendrycksTest-*"]
#TASK_LIST=["wikitext,wiki_zh"]
#TASK_LIST=["ceval-*"]

## lm_eval/adaptor/nemo_eval.py
torchrun --nnodes ${NUM_NODES} --nproc_per_node ${GPUS_PER_NODE} \
    --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --node_rank ${NODE_RANK} \
    ./adaptor/nemo/nemo_eval.py  \
    --config-path=conf \
    --config-name=megatron_gpt_config \
    trainer.devices=${GPUS_PER_NODE} \
    trainer.num_nodes=${NUM_NODES} \
    model.hparams_file=${HPARAM_FILE} \
    model.micro_batch_size=1 \
    model.tensor_model_parallel_size=1 \
    model.pipeline_model_parallel_size=1 \
    model.encoder_seq_length=1024 \
    model.tokenizer.vocab_file=${VOCAB_FILE} \
    model.tokenizer.merge_file=${MERGE_FILE} \
    model.data.gpt_model_file=${NEMO_FILE} \
    model.data.checkpoint_dir=${MODEL_DIR} \
    model.data.checkpoint_name=${MODEL_NAME} \
    model.data.test_data_dir=${TEST_DATA_DIR} \
    model.data.task_list=${TASK_LIST} \
    model.data.num_fewshot=${NUM_FEWSHOT} \
    model.data.max_generate_tokens=${MAX_GENERATE_NUM} \
    model.data.segments=${SEGMENTS} \
    model.data.ans_idx=${ANSIDX} \
    model.data.ratio=${RATIO} \
    model.data.forward_single=${FORWARD_SINGLE} \
