#!/bin/bash

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_DATASETS_OFFLINE=1

readonly GPUS_PER_NODE=8
readonly NODE_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly NNODES="${OMPI_COMM_WORLD_SIZE:-1}"
readonly WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
readonly MASTER_PORT=65535
export MASTER_ADDR="${_MASTER_ADDR:-localhost}"

readonly MODEL_YAML="gpatch/model_yamls/qwen3-30b-a3b-moe-rm.yaml"
readonly DFS_FOLDER="/mnt/gemininjceph2/geminicephfs/mm-base-plt2"
readonly LOAD_CHECKPOINT_DIR="$PWD/qwen3_30b_a3b_moe_mlm"
readonly SAVE_CHECKPOINT_DIR="$PWD/qwen3_30b_a3b_moe_mlm_random_rm_head"
readonly TOKENIZER_MODEL="$DFS_FOLDER/nrwu/hf-hub/Qwen/Qwen3-30B-A3B"

readonly TP_SIZE=1
readonly PP_SIZE=8
readonly EP_SIZE=1

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
    --sequence-parallel \
    --use-distributed-optimizer \
"

TRAINER_ARGS="
    --seq-length 16384 \
    --seed 1111 \
    --eod-mask-loss \
    --micro-batch-size 1 \
    --global-batch-size 1 \
    --train-iters 1 \
    --use-flash-attn \
    --attention-backend flash \
"


EXPERT_ARGS="
    --moe-grouped-gemm \
    --moe-token-dispatcher-type allgather \
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
    --load-model-provider tasks.math_rl_v3.sft \
    --save-model-provider tasks.math_rl_v3.train_ppo_critic \
"

torchrun $DISTRIBUTED_ARGS tools/px_ckpt_conv/convert_sft_to_rm.py \
    $MP_ARGS \
    $TRAINER_ARGS \
    $EXPERT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --cli-arg-yaml-cfgs $MODEL_YAML \
    --distributed-backend nccl \
    --save $SAVE_CHECKPOINT_DIR \
    --load $LOAD_CHECKPOINT_DIR \
