#!/bin/bash

MLM_INPUT_PATH=$1
HF_SAVE_BASE_PATH=$2
readonly MPI_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly MPI_SIZE="${OMPI_COMM_WORLD_SIZE:-1}"
pip install fla-core

PATH_TO_MEGATRON_BRIDGE="../3rdparty/Megatron-Bridge/"
PATH_TO_MEGATRON_DEV="../3rdparty/Megatron-LM/"
PATH_TO_GCORE_DEV="$PWD"
export PYTHONPATH="$PATH_TO_MEGATRON_BRIDGE/src:$PATH_TO_MEGATRON_BRIDGE:$PATH_TO_MEGATRON_DEV:$PATH_TO_GCORE_DEV:$PYTHONPATH"

# Find all iter_* directories and sort them
ITER_DIRS=($(find "$MLM_INPUT_PATH" -maxdepth 1 -type d -name "iter_*" | sort))
TOTAL_ITERS=${#ITER_DIRS[@]}

echo "MPI_RANK: $MPI_RANK, MPI_SIZE: $MPI_SIZE, TOTAL_ITERS: $TOTAL_ITERS"

# Iterate through checkpoints assigned to this rank
for ((i = MPI_RANK; i < TOTAL_ITERS; i += MPI_SIZE)); do
    ITER_DIR=${ITER_DIRS[$i]}
    ITER_NAME=$(basename "$ITER_DIR")
    
    echo "Rank $MPI_RANK processing: $ITER_NAME"
    
    # Create HF save path for this iteration
    HF_SAVE_PATH="${HF_SAVE_BASE_PATH}/${ITER_NAME}"
    
    # Copy config file
    cp examples/astrachang/qwen-next/ugly_bridge_yaml.yaml "$ITER_DIR/run_config.yaml"
    
    # Remove training progress
    python tasks/gpt_oss/rm_training_progress_common.py --mlm-path "$ITER_DIR"
    
    # Convert checkpoint
    python $PATH_TO_MEGATRON_BRIDGE/examples/conversion/convert_checkpoints.py export \
        --hf-model /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_7/user_astrachang/model/Qwen/Qwen3-Next-80B-A3B-Instruct \
        --hf-path "$HF_SAVE_PATH" \
        --megatron-path "$ITER_DIR"
    
    echo "Rank $MPI_RANK completed: $ITER_NAME -> $HF_SAVE_PATH"
done

echo "Rank $MPI_RANK finished all assigned checkpoints."