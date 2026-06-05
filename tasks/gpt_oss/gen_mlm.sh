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

# /mnt/gemininjceph2/geminicephfs/mm-base-plt2/nrwu/hf-hub/Qwen/Qwen3-30B-A3B/
readonly MODEL_YAML="gpatch/model_yamls/gpt-oss-20b.yaml"
readonly DFS_FOLDER="/mnt/ceph-hz1-csp/mm-base-plt2"
readonly LOAD_CHECKPOINT_DIR="/mnt/ceph-hz1-csp/mm-base-plt2/user_astrachang/code/3rdparty/script/gpt-oss-mlm"
readonly SAVE_CHECKPOINT_DIR="$PWD/gpt_oss_20b_mlm"
readonly TOKENIZER_MODEL="$DFS_FOLDER/nrwu/hf-hub/openai/gpt-oss-20b/"
readonly TB_DIR="tb/gpt_oss_20b_mlm"

readonly SEQ_LEN=512
readonly TP_SIZE=1
readonly PP_SIZE=8
readonly EP_SIZE=1
readonly DP_SIZE=$(($GPUS_PER_NODE*$NNODES/$TP_SIZE/$PP_SIZE))
readonly MICRO_BATCH_SIZE=1
readonly GLOBAL_BATCH_SIZE=8


readonly TRAIN_ITERS=5000
readonly EVAL_ITERS=0

echo "INFO
NODE_RANK $NODE_RANK
NNODES $NNODES
TP_SIZE $TP_SIZE
PP_SIZE $PP_SIZE
EP_SIZE $EP_SIZE
DP_SIZE $DP_SIZE
MICRO_BATCH_SIZE $MICRO_BATCH_SIZE
GRADIENT_ACCUMULATE_STEP $GRADIENT_ACCUMULATE_STEP
GLOBAL_BATCH_SIZE $GLOBAL_BATCH_SIZE
"

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
    --expert-model-parallel-size $EP_SIZE \
"


TRAINER_ARGS="
    --seq-length $SEQ_LEN \
    --seed 1111 \
    --eod-mask-loss \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-iters $TRAIN_ITERS \
    --lr 2e-5 \
    --min-lr 0 \
    --lr-warmup-iters 20 \
    --lr-decay-style cosine \
    --weight-decay 0 \
    --clip-grad 1.0 \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --attention-backend flash \
"

DATA_ARGS="
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
"

EXPERT_ARGS="
    --moe-grouped-gemm \
    --moe-token-dispatcher-type allgather \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1 \
    --eval-interval 1 \
    --eval-iters 1 \
    --load-model-provider tasks.gpt_oss.sft \
    --finetune \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
"
# base model 的 prompt
# --prompts 1+1的结果是 \
# 3-4的结果乘以5等于 \
# 5-3>0是对的吗 \

    # --prompts 1+1的结果是2 \
    # 3-4的结果乘以5等于-5 \
    # 5-3>0是对的 \

# 下面是 rm 的
GEN_ARGS="
    --inference-batch-times-seqlen-threshold 4 \
    --max_new_tokens 128 \
    --prompts \"3-4的结果乘以5等于\" \
    \"5-3>0是对的吗\" \
    \"李白，字太白\" \
"

readonly MLM_PATH="/mnt/ceph-hz1-csp/mm-base-plt2/user_astrachang/code/3rdparty/ft_local/Megatron-MoE-EA"

PYTHONPATH="$PWD:$MLM_PATH:$PYTHONPATH" \
torchrun $DISTRIBUTED_ARGS tasks/gpt_oss/generate_mlm.py \
    $TRAINER_ARGS \
    $MP_ARGS \
    $EXPERT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $GEN_ARGS \
    --cli-arg-yaml-cfgs $MODEL_YAML \
    --distributed-backend nccl \
    --save $SAVE_CHECKPOINT_DIR \
    --load $LOAD_CHECKPOINT_DIR
