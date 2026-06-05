#!/bin/bash
# $1: the processing num each node
# $2: the run script
# ${@:3}: the run script param

# ban to use the gpu
export CUDA_VISIBLE_DEVICES=""

# processor
readonly PROCESSOR_PER_NODE=$1
readonly NODE_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly NNODES="${OMPI_COMM_WORLD_SIZE:-1}"
readonly WORLD_SIZE=$(($PROCESSOR_PER_NODE*$NNODES))

readonly MASTER_PORT=61532
export MASTER_ADDR="${_MASTER_ADDR:-localhost}"

echo "INFO
NODE_RANK $NODE_RANK
NNODES $NNODES
PROCESSOR_PER_NODE $PROCESSOR_PER_NODE
WORLD_SIZE $WORLD_SIZE
MASTER_PORT $MASTER_PORT
MASTER_ADDR $MASTER_ADDR
"

DISTRIBUTED_ARGS="
    --nproc_per_node $PROCESSOR_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

readonly MLM_PATH=/root/Megatron-LM
export PYTHONPATH="$MLM_PATH:$PYTHONPATH"

PYTHONPATH="${PWD}:$PYTHONPATH" torchrun $DISTRIBUTED_ARGS $2 ${@:3}
