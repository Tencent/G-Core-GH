#!/bin/bash

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_SOCKET_IFNAME=bond1
export HF_DATASETS_OFFLINE=1

readonly GPUS_PER_NODE=8
readonly NODE_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly NNODES="${OMPI_COMM_WORLD_SIZE:-1}"
readonly WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
readonly MASTER_PORT=65535
export MASTER_ADDR="${_MASTER_ADDR:-localhost}"

readonly DFS_PATH="/mnt/ceph-sz2-csp/mm-base-plt2"
readonly MODEL_YAML="gpatch/model_yamls/qwen3-1.7b-dense.yaml"
readonly DPO_LOAD_CHECKPOINT_DIR="$PWD/qwen3_1-7b_mlm_dpo"
readonly SFT_SAVE_CHECKPOINT_DIR="$PWD/qwen3_1-7b_mlm_converted"
readonly TOKENIZER_MODEL="$DFS_PATH/nrwu/hf-hub/Qwen/Qwen3-1.7B/"

readonly TP_SIZE=2
readonly PP_SIZE=4
readonly SEQ_LEN=4096


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

MP_ARGS="
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --use-distributed-optimizer \
"

TRAINING_ARGS="
    --seq-length $SEQ_LEN \
    --seed 1111 \
    --eod-mask-loss \
    --micro-batch-size 1 \
    --global-batch-size 1 \
    --train-iters 1 \
    --init-method-std 0.02 \
    --use-cpu-initialization \
    --no-initialization \
"

DATA_ARGS="
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
"

OUTPUT_ARGS="
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --log-interval 1 \
    --save-interval 1 \
    --eval-interval 1 \
    --eval-iters 1 \
"

CONVERT_ARGS=" \
"

TASK_ARGS="
    --dpo-model-using both \
    --dpo-policy-ref-model-cnt 2 \
"


RUN_PY="tools/px_ckpt_conv/convert_dpo_to_sft.py"
torchrun $DISTRIBUTED_ARGS $RUN_PY \
    $MP_ARGS \
    $TRAINING_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $TASK_ARGS \
    $CONVERT_ARGS \
    --cli-arg-yaml-cfgs $MODEL_YAML \
    --distributed-backend nccl \
    --save $SFT_SAVE_CHECKPOINT_DIR \
    --load $DPO_LOAD_CHECKPOINT_DIR \
