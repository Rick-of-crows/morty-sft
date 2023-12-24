#!/bin/bash

DATE=`date '+%Y-%m-%d'`

TAG="-debug"

# CHECKPOINT_PATH=/workspace/workspace/yanchao/yanc/bloom-1b1-optimizer-states
# use in docker, need -v when launch docker
# CHECKPOINT_PATH=/workspace/yanchao/yanc/nemo-bloomz
# CHECKPOINT_PATH=/workspace/yanchao/yanc/bloomz-cw/pretrain-hf/bloomz-step276700
CHECKPOINT_PATH=/workspace/BLOOMChat-176B-v1
SAVE_CHECKPOINT_PATH=/workspace/BLOOMChat-Deepspeed-2/global_step0
DATA_PATH=data/cw_total_v13_text_document
TENSORBOARD_PATH=output_dir/tensorboard/$DATE-bloomz-7b1$TAG

mkdir -p $SAVE_CHECKPOINT_PATH
# mkdir -p $TENSORBOARD_PATH

N_GPUS=1
# global-batch-size should divisible by micro-batch-size times data-parallel-size
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1
TP_SIZE=1
PP_SIZE=1

NLAYERS=70
NHIDDEN=14336
NHEADS=112
SEQ_LEN=1024

# llama config in huggingface
# "pad_token_id": -1, "rms_norm_eps": 1e-06,
# "bos_token_id": 0, "eos_token_id": 1, "hidden_act": "silu"
# "vocab_size": 32000


# 1 epoch need iter: 5287827/(2048*16)=161.37 iter
SAVE_INTERVAL=161

# alpaca tokens: 5287827, 10 epoch samples = 5287827/2048*10=25,819
TRAIN_SAMPLES=26_000

LR_DECAY_SAMPLES=23_000
LR_WARMUP_SAMPLES=2000

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.99 \
    --adam-eps 1e-8 \
    --lr 2e-5 \
    --min-lr 1e-6 \
    --lr-decay-style cosine \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
    --clip-grad 1.0 \
    --weight-decay 0. \
    "

# no rampup-batch-size
GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --pp-partition-method type:transformer|embedding \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path /workspace/tokenizer \
    --pad-id 3 \
    --embed-layernorm \
    --fp16 \
    --loss-scale 12 \
    --clip-grad 1.0 \
    --init-method-std 0.0048 \
    --seed 42 \
    --position-embedding-type alibi \
    --checkpoint-activations \
    --pad-vocab-size-to 250880 \
    $OPTIMIZER_ARGS \
    "

OUTPUT_ARGS=" \
    --exit-interval 5000 \
    --log-interval 1 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 50 \
    --eval-iters 1 \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "

#     --tensorboard-dir $TENSORBOARD_PATH \
<<!
ZERO_STAGE=0

config_json="./ds_config.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT
!

ZERO_STAGE=0

config_json="./ds_config.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "bf16": {
    "enabled": true
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT


DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    --deepspeed-activation-checkpointing \
    "

CUSTOM_ARGS=" \
    --no-load-optim \
    --finetune \
    --override-lr-scheduler \
    --no-save-rng \
    --inference \
    "

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export LAUNCHER="python -u -m torch.distributed.launch \
    --nproc_per_node $N_GPUS \
    --nnodes 1 \
    --master_port 60400 \
    "
    
export CMD=" \
    $LAUNCHER run_check_bloom_ds_and_hf.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --bloom-hf-ckpt $CHECKPOINT_PATH \
    --save $SAVE_CHECKPOINT_PATH \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    $CUSTOM_ARGS \
    --data-path $DATA_PATH \
    --data-impl mmap \
    --split 949,50,1 \
    --kill-switch-path /tmp/kill-switch \
    --distributed-backend nccl \
    $DEEPSPEED_ARGS \
    "

# do not remove or the training will hang and nodes will be lost w/o this workaround
export CUDA_LAUNCH_BLOCKING=1

# hide duplicated errors using this hack - will be properly fixed in pt-1.12
export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1

echo $CMD

$CMD  #  2>&1 |tee pretrain_gpt.bloom-1b1.$DATE$TAG.log
