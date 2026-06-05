#!/bin/bash
# pip install -U transformers
# pip install fla-core
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

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_DATASETS_OFFLINE=1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_SOCKET_IFNAME=bond1
export CUDNN_PATH=/usr/lib64
# export NVTE_DEBUG=1
# export NVTE_DEBUG_LEVEL=2
# export CUDNN_LOGERR_DBG=1
# export CUDNN_LOGDEST_DBG=stderr

readonly GPUS_PER_NODE=8
readonly NODE_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly NNODES="${OMPI_COMM_WORLD_SIZE:-1}"
readonly WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
readonly MASTER_PORT=65535
export MASTER_ADDR="${_MASTER_ADDR:-localhost}"

readonly DFS_FOLDER=$PWD
readonly HF_HUB_DIR="/mnt/ceph-hz1-csp/mm-base-plt2/user_rionawang/project/models/Qwen3-Coder-Next"
readonly LOAD_CHECKPOINT_DIR="$HF_HUB_DIR"
readonly SAVE_CHECKPOINT_DIR="qwen3_next_coder_80b_a3b_sft_save"
readonly TOKENIZER_MODEL="$HF_HUB_DIR"
readonly MLM_PATH="/root/Megatron-LM"
readonly MBRIDGE_PATH="/root/Megatron-Bridge"
export PYTHONPATH="$MLM_PATH:$MBRIDGE_PATH/src:$PWD:$PYTHONPATH"

readonly TP_SIZE=2
readonly PP_SIZE=1
readonly CP_SIZE=1
readonly EP_SIZE=32
readonly MICRO_BATCH_SIZE=1
readonly GLOBAL_BATCH_SIZE=512
readonly TRAIN_ITERS=2737
readonly SEQ_LENGTH=$((4*1024))

echo "INFO
NODE_RANK $NODE_RANK
NNODES $NNODES
TP_SIZE $TP_SIZE
PP_SIZE $PP_SIZE
CP_SIZE $CP_SIZE
EP_SIZE $EP_SIZE
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
    --expert-tensor-parallel-size 1 \
    --expert-model-parallel-size $EP_SIZE \
    --sequence-parallel \
    --context-parallel-size $CP_SIZE \
    --use-distributed-optimizer \
    --attention-backend auto \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --recompute-granularity full \
"

TRAINER_ARGS="
    --seq-length $SEQ_LENGTH \
    --seed 1111 \
    --eod-mask-loss \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-iters $TRAIN_ITERS \
    --lr 8e-6 \
    --min-lr 0 \
    --lr-warmup-iters 200 \
    --lr-decay-style cosine \
    --optimizer adam \
    --weight-decay 0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --sft \
    --cp-comm-type a2a \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 0.001 \
    --moe-pad-with-random-token \
    --try-load-from-megatron-bridge \
    --export-to-hf-megatron-bridge \
    --megatron-bridge-distributed-filesystem \
"

DATA_ARGS="
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --dataloader-type external \
    --num-workers 1 \
    --px-data-config-path tasks/math_rl_v3/qwen/sft_data_config.json \
    --px-shuffle-data \
    --px-shuffle-buffer-size 102400 \
    --px-use-indexed-jsonl-dataset \
    --px-auto-cal-eval-iters \
"

export WANDB_BASE_URL=
export WANDB_API_KEY=

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --tensorboard-dir tb/sft_qwen_coder \
    --tensorboard-log-interval 1 \
    --eval-interval 9999999999 \
    --eval-iters 1 \
    --wandb-project qwen3_coder_next_riona \
    --wandb-exp-name qwen-coder-next-sft-tp$TP_SIZE-pp$PP_SIZE-cp$CP_SIZE-ep$EP_SIZE \
    --wandb-save-dir wandb \
"

# 热启动之后去掉这些 flag
FINETUNE_ARGS="
    --finetune \
    --no-load-optim \
    --no-load-rng \
"

torchrun $DISTRIBUTED_ARGS tasks/math_rl_v3/qwen3/moe/coder_next/sft_qwen3_coder_next.py \
    $TRAINER_ARGS \
    $MP_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $FINETUNE_ARGS \
    --distributed-backend nccl \
    --px-apply-chat-template \
    --cli-arg-yaml-cfgs gpatch/model_yamls/qwen3-coder-next-80b.yaml \
    --save $SAVE_CHECKPOINT_DIR \
    --load $LOAD_CHECKPOINT_DIR \
    --hf-model-path ${HF_HUB_DIR}
