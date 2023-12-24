#!/bin/bash

DATE=`date '+%Y-%m-%d'`

TAG="-cw-padding-v15-0612"

# CHECKPOINT_PATH=/workspace/workspace/yanchao/yanc/bloom-1b1-optimizer-states
# use in docker, need -v when launch docker
#CHECKPOINT_PATH=/workspace/2023-05-31/bloomz-7b-cw-pretrain-v5-sft
#CHECKPOINT_PATH=/workspace/2023-06-09/bloomz-7b-cw-pretrain-v5-0609
CHECKPOINT_PATH=/workspace/yanc-share/2023-05-29/bloomz-7b-cw-pretrain-v5-math
SAVE_CHECKPOINT_PATH=/workspace/yanc-share/$DATE/bloomz-7b-cw-padding-v15-0612
DATA_PATH=data/cw/cw_total_v15_text_document
#DATA_PATH=data/pretrain/cw-total_text_document
TENSORBOARD_PATH=output_dir/tensorboard/$DATE-bloom-7b$TAG

mkdir -p $SAVE_CHECKPOINT_PATH
mkdir -p $TENSORBOARD_PATH

N_GPUS=8
# global-batch-size should divisible by micro-batch-size times data-parallel-size
MICRO_BATCH_SIZE=8
GLOBAL_BATCH_SIZE=128
# NGPU = TP_SIZE*PP_SIZE*DP_SIZE
TP_SIZE=2
PP_SIZE=1

NLAYERS=30
NHIDDEN=4096
NHEADS=32
SEQ_LEN=2048

# llama config in huggingface
# "pad_token_id": -1, "rms_norm_eps": 1e-06,
# "bos_token_id": 0, "eos_token_id": 1, "hidden_act": "silu"
# "vocab_size": 32000


# 1 epoch need iter: 5287827/(2048*16)=161.37 iter
# This need change when continue training
#2924 /iter
SAVE_INTERVAL=1687

# alpaca tokens: 5287827, 10 epoch samples = 5287827/2048*10=25,819

TRAIN_ITERS=16870 #source iter
#use old
LR_DECAY_ITERS=16026
LR_WARMUP_ITERS=506 # use old

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.99 \
    --adam-eps 1e-8 \
    --lr 2e-5 \
    --min-lr 1e-6 \
    --clip-grad 1.0 \
    --lr-decay-style cosine \
    --lr-decay-iters $LR_DECAY_ITERS \
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
    --pp-partition-method type:transformer|embedding \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path /workspace/yanc/tokenizer \
    --embed-layernorm \
    --pad-id 3 \
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
 #--use-dynamic-padding \
OUTPUT_ARGS=" \
    --exit-interval 50000 \
    --log-interval 1 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 600 \
    --eval-iters 1 \
    --train-iters=$TRAIN_ITERS
    --lr-warmup-iters=$LR_WARMUP_ITERS
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "

ZERO_STAGE=1

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
    "

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

<<!
echo "Pass......................"

export LAUNCHER="python -u -m torch.distributed.launch \
    --nproc_per_node $N_GPUS \
    --nnodes 1 \
    --master_port 60400 \
    "
!

export LAUNCHER="deepspeed --hostfile=hostfile --no_ssh_check"
export CMD=" \
    $LAUNCHER pretrain_gpt_no_concaten.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    $CUSTOM_ARGS \
    --save $SAVE_CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --data-impl mmap \
    --split 995,4,1 \
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
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8
export NCCL_IB_TIMEOUT=23
export NCCL_IB_GID_INDEX=3   # 走roce v2 协议

echo $CMD

$CMD  2>&1 |tee pretrain_gpt.bloomz-7b1.$DATE$TAG.log
