#!/bin/bash
# ban to use the gpu
export CUDA_VISIBLE_DEVICES=""

PLACE_CFG_FOLDER=$1
MYWD=$PWD
# 理论上要改 HF_HUB_DIR、DATASET_TYPE（特别要注意sft不能使用 grpo）、RULE_TYPE
# readonly HF_HUB_DIR="$MYWD/hf-hub/Qwen/Qwen2.5-VL-3B-Instruct"
readonly HF_HUB_DIR="$MYWD/hf-hub/Qwen/Qwen3-VL-30B-A3B-Instruct"

# readonly DATASET_ROOT="${MYWD}/hf-hub/RadGenome/PMC-VQA/gcore-data"
# JSONL_PATH="${DATASET_ROOT}/filter_4k/test_2.csv.jsonl"
# DATASET_TYPE="sft"
# RULE_TYPE="captcha" # 可以不填

# readonly DATASET_ROOT="${MYWD}/hf-hub/hiyouga/geometry3k/gcore-data"
# JSONL_PATH="${DATASET_ROOT}/filter_4k/test-00000-of-00001.jsonl"
# DATASET_TYPE="grpo"
# RULE_TYPE="geometry3k"

readonly DATASET_ROOT="${MYWD}/hf-hub/hiyouga/geometry3k/gcore-data-v2"
JSONL_PATH="${DATASET_ROOT}/test/test-00000-of-00001.jsonl"
DATASET_TYPE="grpo"
RULE_TYPE="geometry3k-v2"

# readonly DATASET_ROOT="${MYWD}/hf-hub/RadGenome/PMC-VQA/gcore-data-grpo"
# JSONL_PATH="${DATASET_ROOT}/test/test_2.csv.jsonl"
# DATASET_TYPE="grpo"
# RULE_TYPE="pmc_vqa"

readonly LMDB_PATH="${DATASET_ROOT}/img_file.lmdb"
EVAL_CFG_PATH="${PLACE_CFG_FOLDER}/config.json"
LMDB_PORT=8312

# processor
readonly PROCESSOR_PER_NODE=64
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

readonly MLM_PATH=../Megatron-LM
export PYTHONPATH="$MLM_PATH:$PYTHONPATH"

pkill -f -9 lmdb_read_svr.py
nohup python megatron_datasets/tools/lmdb_read_svr.py \
    --lmdb-path $LMDB_PATH \
    --lmdb-map-size 500 \
    --lmdb-port $LMDB_PORT > svr.log 2>&1 &
sleep 3


torchrun $DISTRIBUTED_ARGS  tools/eval/eval_client.py \
        --model-path ${HF_HUB_DIR} \
        --config ${EVAL_CFG_PATH} \
        --jsonl-path ${JSONL_PATH} \
        --lmdb-port ${LMDB_PORT} \
        --dataset-type ${DATASET_TYPE} \
        --ppo-mm-rule-type ${RULE_TYPE} \
        --max-tokens 2048

pkill -f -9 lmdb_read_svr.py
