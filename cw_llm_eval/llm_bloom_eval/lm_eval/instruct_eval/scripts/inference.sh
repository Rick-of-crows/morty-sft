MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=29500

export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3

NUM_NODES=2
GPUS_PER_NODE=8

DATA_PATH=outputs/fast_data.json
CONFIG=configs/test.yml
PLAYER=player1
OUTPUT=outputs/
BATCHSIZE=4

torchrun --nnodes ${NUM_NODES} --nproc_per_node ${GPUS_PER_NODE} \
    --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --node_rank ${NODE_RANK} \
    player.py \
    -data ${DATA_PATH} \
    -config ${CONFIG} \
    -player ${PLAYER} \
    -output ${OUTPUT} \
    -bs ${BATCHSIZE}
