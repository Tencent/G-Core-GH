#!/bin/bash

# 每台机器的 GPU 数量
readonly GPU_PER_NODE=8
readonly NODE_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly NNODES="${OMPI_COMM_WORLD_SIZE:-1}"

readonly MASTER_PORT=61533
export MASTER_ADDR="${_MASTER_ADDR:-localhost}"

echo "INFO
NODE_RANK $NODE_RANK
NNODES $NNODES
GPU_PER_NODE $GPU_PER_NODE
MASTER_PORT $MASTER_PORT
MASTER_ADDR $MASTER_ADDR
"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPU_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

torchrun $DISTRIBUTED_ARGS tools/check_gpu/check_gpu.py
