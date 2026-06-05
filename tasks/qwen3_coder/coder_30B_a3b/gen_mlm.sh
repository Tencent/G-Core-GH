#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_DATASETS_OFFLINE=1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_SOCKET_IFNAME=bond1

readonly GPUS_PER_NODE=8
readonly NODE_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly NNODES="${OMPI_COMM_WORLD_SIZE:-1}"
readonly WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
readonly MASTER_PORT=65535
export MASTER_ADDR="${_MASTER_ADDR:-localhost}"

# /mnt/gemininjceph2/geminicephfs/mm-base-plt2/nrwu/hf-hub/Qwen/Qwen3-30B-A3B/
readonly MODEL_YAML="gpatch/model_yamls/qwen3-coder-30b-a3b-instruct.yaml"
readonly DFS_FOLDER="/mnt/ceph-hz1-csp/mm-base-plt2/"
readonly LOAD_CHECKPOINT_DIR="$PWD/qwen3_coder_30b_a3b_moe_mlm"
readonly SAVE_CHECKPOINT_DIR="$PWD/qwen3_coder_30b_a3b_moe_mlm_save"
readonly TOKENIZER_MODEL="$PWD/hf-hub/Qwen/Qwen3-Coder-30B-A3B-Instruct/"
readonly TB_DIR="tb/qwen3_coder_30b_a3b_moe"

readonly SEQ_LEN=512
readonly TP_SIZE=1
readonly PP_SIZE=1
readonly EP_SIZE=8
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
    --expert-tensor-parallel-size 1 \
"


TRAINER_ARGS="
    --use-mcore-models \
    --seq-length $SEQ_LEN \
    --seed 1111 \
    --eod-mask-loss \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --no-bias-swiglu-fusion \
    --no-check-for-nan-in-loss-and-grad \
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

# --load-model-provider gpatch.training.v3.default_model_provider_0_13 \
OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1 \
    --eval-interval 1 \
    --eval-iters 1 \
    --load-model-provider gpatch.training.v3.default_model_provider \
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
    --max_new_tokens 256 \
    --prompts 5-3>0是对的吗 \
"

export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_ZERO_CENTERED_GAMMA_IN_WTYPE=1


MLM_PATH="../3rdparty/Megatron-LM/"
export PYTHONPATH="$PWD:$MLM_PATH:$PYTHONPATH"
torchrun $DISTRIBUTED_ARGS tools/align_loss/generate_mlm.py \
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
