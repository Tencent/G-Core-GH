#!/bin/bash

ps -ef | grep python | awk  '{print $2}' | xargs -I {} kill -9 {}
sleep 1

set -ex

TS=$(date +%Y-%m-%d-%H-%M-%S)

# # DeepEP env begin
# export NCCL_SOCKET_IFNAME=bond1
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_HCA=mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1,mlx5_bond_4:1,mlx5_bond_5:1,mlx5_bond_6:1,mlx5_bond_7:1,mlx5_bond_8:1

# export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond1
# export NVSHMEM_HCA_LIST=mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1,mlx5_bond_4:1,mlx5_bond_5:1,mlx5_bond_6:1,mlx5_bond_7:1,mlx5_bond_8:1

# export NCCL_IB_TC=160
# export NVSHMEM_IB_TRAFFIC_CLASS=160
# # DeeepEP env end

# env from model zoo begin
export TORCH_NCCL_AVOID_RECORD_STREAMS=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_NVLS_ENABLE=0
export NVTE_FUSED_ATTN=1
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1
export PYTHONWARNINGS=ignore
export NCCL_DEBUG=VERSION
# env from model zoo end

# export PYTHONPATH="$PWD:../Megatron-LM-main:$PYTHONPATH"
# export PYTHONPATH="$PWD:../Megatron-LM-core_r0.13:$PYTHONPATH"
# export PYTHONPATH="$PWD:../Megatron-LM:$PYTHONPATH"
export PYTHONPATH="$PWD:../Megatron-LM:$PYTHONPATH"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_DATASETS_OFFLINE=1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_SOCKET_IFNAME=bond1
export NVTE_FUSED_ATTN=1

readonly GPUS_PER_NODE=8
readonly NODE_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly NNODES="${OMPI_COMM_WORLD_SIZE:-1}"
readonly WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
readonly MASTER_PORT=65535
export MASTER_ADDR="${_MASTER_ADDR:-localhost}"

# readonly DFS_FOLDER="/mnt/gemininjceph2/geminicephfs/mm-base-plt2"
readonly DFS_FOLDER='/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_7'
readonly LOAD_CHECKPOINT_DIR="$PWD/deepseek-v3-mlm-ckpt"
readonly SAVE_CHECKPOINT_DIR="$PWD/deepseek-v3-sft-bf16"
readonly TOKENIZER_MODEL="$DFS_FOLDER/user_jeffhong/hf-hub/deepseek-ai/DeepSeek-V3"

readonly EXP_NAME="deepseek-v3-mlm-align"
readonly TB_DIR="tb/deepseek-v3-mlm-align"
readonly WANDB_DIR="wandb-save/deepseek-v3-mlm-align"

readonly TP_SIZE=8
readonly PP_SIZE=8
readonly CP_SIZE=4
readonly EP_SIZE=16
readonly ETP_SIZE=1
readonly MICRO_BATCH_SIZE=1
readonly GLOBAL_BATCH_SIZE=256

readonly TRAIN_ITERS=3357
readonly EVAL_ITERS=0
readonly SEQ_LENGTH=25600

echo "INFO
NODE_RANK $NODE_RANK
NNODES $NNODES
TP_SIZE $TP_SIZE
PP_SIZE $PP_SIZE
CP_SIZE $CP_SIZE
EP_SIZE $EP_SIZE
ETP_SIZE $ETP_SIZE
MICRO_BATCH_SIZE $MICRO_BATCH_SIZE
GRADIENT_ACCUMULATE_STEP $GRADIENT_ACCUMULATE_STEP
GLOBAL_BATCH_SIZE $GLOBAL_BATCH_SIZE
"

# torch 启动参数
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

# 并行度
    # --decoder-first-pipeline-num-layers 15 \
    # --decoder-last-pipeline-num-layers 14 \
MP_ARGS="
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --context-parallel-size $CP_SIZE \
    --expert-model-parallel-size $EP_SIZE \
    --expert-tensor-parallel-size $ETP_SIZE \
    --use-distributed-optimizer \
    --attention-backend fused \
"

# export PP_LAYOUT="Et*3|(tt|)*29|L"
if [[ ${PP_SIZE} -gt 1 ]]; then
    if [[ -n ${PP_LAYOUT} ]]; then
        MP_ARGS="${MP_ARGS} --pipeline-model-parallel-layout ${PP_LAYOUT} "
    elif [[ ${PP_SIZE} -eq 4 ]]; then
        MP_ARGS="${MP_ARGS} --decoder-first-pipeline-num-layers 15 --decoder-last-pipeline-num-layers 14 "
    elif [[ ${PP_SIZE} -eq 8 ]]; then
        MP_ARGS="${MP_ARGS} --decoder-first-pipeline-num-layers 6 --decoder-last-pipeline-num-layers 7 "
    elif [[ ${PP_SIZE} -eq 16 ]]; then
        MP_ARGS="${MP_ARGS} --decoder-first-pipeline-num-layers 3 --decoder-last-pipeline-num-layers 2 "
    fi
fi

readonly MODEL_YAML="gpatch/model_yamls/deepseek-v3.yaml"

TRAINER_ARGS="
    --seq-length $SEQ_LENGTH \
    --sequence-parallel \
    --use-flash-attn \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --no-check-for-nan-in-loss-and-grad \
    --no-save-optim \
    --manual-gc \
    --manual-gc-interval 10 \
"

OPTIMIZER_ARGS="
    --lr 0 \
    --min-lr 0 \
    --lr-decay-style cosine \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --train-iters ${TRAIN_ITERS} \
"

# 数据与 tokenizer
DATA_ARGS="
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --dataloader-type external \
    --num-workers 1 \
    --px-data-config-path tasks/math_rl_v3/qwen/sft_data_config.json \
    --px-use-indexed-jsonl-dataset \
    --px-auto-cal-eval-iters \
"

# --px-shuffle-data \
# --px-shuffle-buffer-size 10000 \

# export WANDB_BASE_URL=
# export WANDB_API_KEY=
export WANDB_BASE_URL=
export WANDB_API_KEY=

# 热启动之后去掉这些 flag
FINETUNE_ARGS="
    --finetune \
    --no-load-optim \
    --no-load-rng \
"

# A2A_OVERLAP=${A2A_OVERLAP:-0}
A2A_OVERLAP=0
if [[ ${A2A_OVERLAP} == 1 ]]; then
    export CUDA_DEVICE_MAX_CONNECTIONS=32
    export NVTE_FWD_LAYERNORM_SM_MARGIN=20
    export NVTE_BWD_LAYERNORM_SM_MARGIN=20
    TRAINER_ARGS="${TRAINER_ARGS} --delay-wgrad-compute --overlap-moe-expert-parallel-comm "
else
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NVTE_FWD_LAYERNORM_SM_MARGIN=0
    export NVTE_BWD_LAYERNORM_SM_MARGIN=0
    TRAINER_ARGS="${TRAINER_ARGS} --overlap-grad-reduce --overlap-param-gather "
fi

# FP8 arguments
PR="bf16"
# PR="fp8"
if [[ ${PR} == "fp8" ]]; then
    TRAINER_ARGS="${TRAINER_ARGS} --fp8-recipe blockwise --fp8-format e4m3"
    # TRAINER_ARGS="${TRAINER_ARGS} --fp8-param-gather" # Optimizer CPU offload does not support fp8 param gather now.
    # TRAINER_ARGS="${TRAINER_ARGS} --load-main-params-from-ckpt "
    TRAINER_ARGS="${TRAINER_ARGS} --use-precision-aware-optimizer --main-grads-dtype fp32 --main-params-dtype fp32 --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16"
    TRAINER_ARGS="${TRAINER_ARGS} --moe-router-padding-for-fp8"
fi

EVAL_AND_OUTPUT_ARGS="
    --eval-interval 200 \
    --eval-iters 1 \
    --log-interval 1 \
    --log-throughput \
    --log-timers-to-tensorboard \
    --log-memory-to-tensorboard \
    --save-interval 200 \
    --tensorboard-dir tb/sft \
    --tensorboard-log-interval 1 \
    --wandb-project jeffhong \
    --wandb-exp-name $TS/deepseek-v3-math-sft-seqlen-${SEQ_LENGTH}-PR-${PR}-tp-${TP_SIZE}-pp-${PP_SIZE}-cp-${CP_SIZE}-ep-${EP_SIZE}-lr-${LEARNING_RATE}-mbs-${MICRO_BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE} \
    --wandb-save-dir wandb \
"

# export CUDNN_LOGERR_DBG=1
# export CUDNN_LOGDEST_DBG=stderr
# export NVTE_DEBUG=1
# export NVTE_DEBUG_LEVEL=2

RUN_PY='tasks/math_rl_v3/sft.py'
# export PX_DEBUG_TRAIN_LOG=1
    # --recompute-granularity selective \
    # --recompute-modules mla_up_proj mlp \
    # --enable-experimental \
torchrun $DISTRIBUTED_ARGS $RUN_PY \
    --seed 1111 \
    --eod-mask-loss \
    $MP_ARGS \
    $DATA_ARGS \
    $TRAINER_ARGS \
    $OPTIMIZER_ARGS \
    $EVAL_AND_OUTPUT_ARGS \
    $FINETUNE_ARGS \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --recompute-granularity full \
    --distributed-backend nccl \
    --px-apply-chat-template \
    --px-system-prompt "$system_prompt" \
    --cli-arg-yaml-cfgs $MODEL_YAML \
    --enable-experimental \
    --save $SAVE_CHECKPOINT_DIR \
    --load $LOAD_CHECKPOINT_DIR >log/math-ds-v3-${NODE_RANK}.log 2>&1 &