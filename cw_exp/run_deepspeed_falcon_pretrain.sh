#!/bin/bash

DATE=`date '+%Y-%m-%d'`
HOUR=`date '+%H-%M-%S'`

# TAG="-debug-no-concaten-loss-valid-hf-to-ds-no-init-load-all"
TAG="40b"

# use in docker, need -v when launch docker
# CHECKPOINT_PATH=checkpoints/test_bloom_to_deepspeed_no_initialize_load_all
# CHECKPOINT_PATH=checkpoints/test_bloom_to_deepspeed
CHECKPOINT_PATH=/workspace/yanc/models/falcon/falcon_step7000_ds
SAVE_CHECKPOINT_PATH=/workspace/yanc/$DATE/falcon-$TAG
# DATA_PATH=data/alpaca_belle-instruct/alpaca_belle-instruct_text_document
DATA_PATH=data/cw/cw_pretrain_text_document
TENSORBOARD_PATH=output_dir/tensorboard/$DATE-falcon$TAG

mkdir -p $SAVE_CHECKPOINT_PATH
mkdir -p $TENSORBOARD_PATH

N_GPUS=8
# global-batch-size should divisible by micro-batch-size times data-parallel-size
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=128
TP_SIZE=4
PP_SIZE=2

NLAYERS=60
NHIDDEN=8192
NHEADS=128
SEQ_LEN=2048

# 1 epoch need iter: 1126713*0.95/16=66898
# 1 epoch need iter: 1126713*0.95/128=8362
SAVE_INTERVAL=4026

# 3 epoch
TRAIN_ITERS=24156

LR_DECAY_ITERS=22948
LR_WARMUP_ITERS=725

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.99 \
    --adam-eps 1e-8 \
    --lr 2e-5 \
    --min-lr 1e-6 \
    --lr-decay-style cosine \
    --lr-decay-iters $LR_DECAY_ITERS \
    --lr-warmup-iters $LR_WARMUP_ITERS \
    --clip-grad 1.0 \
    --weight-decay 0. \
    "

# no rampup-batch-size
GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-iters $TRAIN_ITERS \
    --pp-partition-method type:transformer|embedding \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path /workspace/yanc/models/falcon/falcon_step7000_ds/tokenizer \
    --pad-id 3 \
    --bf16 \
    --clip-grad 1.0 \
    --init-method-std 0.0048 \
    --seed 42 \
    --position-embedding-type rotary \
    --checkpoint-activations \
    --pad-vocab-size-to 65024 \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --n-head-kv 8 \
    $OPTIMIZER_ARGS \
    "
#     --use-dynamic-padding \

OUTPUT_ARGS=" \
    --exit-interval 50000 \
    --log-interval 1 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 600 \
    --eval-iters 1 \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "

ZERO_STAGE=0
config_json="./ds_config.json"

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
    $LAUNCHER pretrain_gpt_falcon.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --save $SAVE_CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    $CUSTOM_ARGS \
    --data-path $DATA_PATH \
    --data-impl mmap \
    --split 995,4,1 \
    --kill-switch-path /tmp/kill-switch \
    --distributed-backend nccl \
    $DEEPSPEED_ARGS \
    "
# do not remove or the training will hang and nodes will be lost w/o this workaround
export CUDA_LAUNCH_BLOCKING=1

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1


# hide duplicated errors using this hack - will be properly fixed in pt-1.12
export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

#export NCCL_DEBUG=INFO
#export NCCL_BUFFERSIZE=81943040
# yanc not work
#export NCCL_P2P_LEVEL=NVL
#export NCCL_P2P_DISABLE=1
#export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ens22f0np0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_6,mlx5_7
#export NCCL_SOCKET_IFNAME=eth1,eth2,eth3,eth4,eth5,eth6,eth7,eth8
#export NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8
#export NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NCCL_IB_TIMEOUT=23
export NCCL_IB_GID_INDEX=3   # 走roce v2 协议
#export NCCL_IB_TC=128


echo $CMD

$CMD  2>&1 |tee pretrain_gpt.falcon-40b.$DATE-$HOUR-$TAG.log
