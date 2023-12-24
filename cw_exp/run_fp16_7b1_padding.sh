#!/bin/bash

DATE=`date '+%Y-%m-%d'`

TAG="-causal-no-concaten-all-loss"

# CHECKPOINT_PATH=/workspace/workspace/yanchao/yanc/bloom-1b1-optimizer-states
# use in docker, need -v when launch docker
CHECKPOINT_PATH=/workspace/bloomz-7b1-optimizer-states
SAVE_CHECKPOINT_PATH=output_dir/checkpoints/$DATE/bloomz-7b1$TAG
DATA_PATH=data/alpaca_belle_cot_suffle/alpaca_belle_cot_suffle_text_document 
TENSORBOARD_PATH=output_dir/tensorboard/$DATE-bloomz-7b1$TAG

mkdir -p $SAVE_CHECKPOINT_PATH
mkdir -p $TENSORBOARD_PATH

N_GPUS=8
# global-batch-size should divisible by micro-batch-size times data-parallel-size
MICRO_BATCH_SIZE=16
GLOBAL_BATCH_SIZE=128
TP_SIZE=2
PP_SIZE=1

NLAYERS=30
NHIDDEN=4096
NHEADS=32
SEQ_LEN=1024

# 1 epoch need iter: 5287827/(2048*16)=161.37 iter
# 1 epoch iter=52002*0.95/128=386
SAVE_INTERVAL=4400

# alpaca tokens: 5287827, 10 epoch samples = 5287827/2048*10=25,819
# TRAIN_SAMPLES=26_000

# LR_DECAY_SAMPLES=23_000
# LR_WARMUP_SAMPLES=2000

# 1 epoch iter=52002*0.95/16=3086.6
# instead of use train-samples, use train-iters to control training steps
# equal to train-samples = global-batch-size * train-iters
TRAIN_ITERS=26_400
LR_DECAY_ITERS=25_000
LR_WARMUP_ITERS=800

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

    # --lr-decay-samples $LR_DECAY_SAMPLES \
    # --lr-warmup-samples $LR_WARMUP_SAMPLES \

# no rampup-batch-size
GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-iters $TRAIN_ITERS \
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

#    --train-samples $TRAIN_SAMPLES \

OUTPUT_ARGS=" \
    --exit-interval 30000 \
    --log-interval 1 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 300 \
    --eval-iters 1 \
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

# no concatenate only need mask row
#     --reset-attention-mask \

# try diff
#     --loss-on-targets-only \

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export LAUNCHER="python -u -m torch.distributed.launch \
    --nproc_per_node $N_GPUS \
    --nnodes 1 \
    --master_port 60400 \
    "
    
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

$CMD  2>&1 |tee pretrain_gpt.bloomz-7b1.$DATE$TAG.log
