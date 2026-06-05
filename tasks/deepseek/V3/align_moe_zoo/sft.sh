#!/bin/bash

ps -ef | grep python | awk  '{print $2}' | xargs -I {} kill -9 {}
sleep 1

set -ex

# DeepEP env begin
export NCCL_SOCKET_IFNAME=bond1
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1,mlx5_bond_4:1,mlx5_bond_5:1,mlx5_bond_6:1,mlx5_bond_7:1,mlx5_bond_8:1

export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond1
export NVSHMEM_HCA_LIST=mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1,mlx5_bond_4:1,mlx5_bond_5:1,mlx5_bond_6:1,mlx5_bond_7:1,mlx5_bond_8:1

export NCCL_IB_TC=160
export NVSHMEM_IB_TRAFFIC_CLASS=160
# DeeepEP env end

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

export PYTHONPATH="$PWD:../Megatron-LM-core_r0.13:$PYTHONPATH"
# export PYTHONPATH="$PWD:../Megatron-LM:$PYTHONPATH"
# export PYTHONPATH="$PWD:../Megatron-LM-main:$PYTHONPATH"

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

readonly DFS_FOLDER='/mnt/ceph-sh2'
# readonly LOAD_CHECKPOINT_DIR="$PWD/xxx"
readonly LOAD_CHECKPOINT_DIR="$PWD/deepseek-v3"
# readonly LOAD_CHECKPOINT_DIR="/mnt/ceph-sh2/user_jeffhong/workspace/wepsdl-dev/Megatron-MoE-ModelZoo/deepseek-v3-dist/torch_dist"
readonly SAVE_CHECKPOINT_DIR="$PWD/deepseek-v3-sft"
readonly TOKENIZER_MODEL="$DFS_FOLDER/user_jeffhong/hf-hub/deepseek-ai/DeepSeek-V3"

readonly TP_SIZE=2
readonly PP_SIZE=8
readonly CP_SIZE=1
readonly EP_SIZE=16
readonly ETP_SIZE=1
readonly MICRO_BATCH_SIZE=1
readonly GLOBAL_BATCH_SIZE=4096
# readonly SEQ_LENGTH=$((4*1024))
readonly SEQ_LENGTH=$((2048))

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

export PP_LAYOUT="Et*3|(tt|)*29|L"
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
    --weight-decay 0.1 \
    --lr-decay-samples 584765624 \
    --lr-warmup-samples 1536000 \
    --lr-warmup-init 3.9e-7 \
    --lr 3.9e-6 \
    --min-lr 3.9e-7 \
    --lr-decay-style cosine \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
"

DATASET_PATH="/mnt/ceph-sh2/user_jeffhong/datasets"

# 数据与 tokenizer
DATA_ARGS="
    --seq-length $SEQ_LENGTH \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --vocab-file $DATASET_PATH/oscar/gpt2-vocab.json \
    --merge-file $DATASET_PATH/oscar/gpt2-merges.txt \
    --data-path $DATASET_PATH/oscar-mcore/oscardata_text_document \
    --split 99,1,0 \
    --no-mmap-bin-files \
    --no-create-attention-mask-in-dataloader \
    --num-workers 6 \
    --train-samples 585937500 \
"
    # --train-samples 79000 \

export WANDB_BASE_URL=
export WANDB_API_KEY=

TS=$(date +%Y-%m-%d-%H-%M-%S)

EVAL_AND_OUTPUT_ARGS="
    --eval-interval 200 \
    --eval-iters 32 \
    --log-interval 1 \
    --log-throughput \
    --save-interval 500 \
    --tensorboard-dir tb/sft \
    --tensorboard-log-interval 1 \
    --wandb-project jeffhong \
    --wandb-exp-name $TS/deepseek-v3-seqlen-${SEQ_LENGTH}-tp-${TP_SIZE}-pp-${PP_SIZE}-cp-${CP_SIZE}-ep-${EP_SIZE}-etp-${ETP_SIZE} \
    --wandb-save-dir wandb \
"

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
    TRAINER_ARGS="${TRAINER_ARGS} --delay-wgrad-compute --overlap-moe-expert-parallel-comm"
else
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NVTE_FWD_LAYERNORM_SM_MARGIN=0
    export NVTE_BWD_LAYERNORM_SM_MARGIN=0
    TRAINER_ARGS="${TRAINER_ARGS} --overlap-grad-reduce --overlap-param-gather "
fi

export PYTHONPATH="$MLM_PATH:$PYTHONPATH"

torchrun $DISTRIBUTED_ARGS tasks/deepseek/V3/align_moe_zoo/sft.py \
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
    --enable-experimental \
    --cli-arg-yaml-cfgs $MODEL_YAML \
    --save $SAVE_CHECKPOINT_DIR \
    --load $LOAD_CHECKPOINT_DIR >log/mlm-sft-oscar-ds-v3-${NODE_RANK}.log 2>&1 &