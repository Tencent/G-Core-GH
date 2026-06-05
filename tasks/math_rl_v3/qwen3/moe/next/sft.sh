#!/bin/bash
pip install -U transformers
pip install fla-core

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
readonly LOAD_CHECKPOINT_DIR="/mnt/geminihzceph1/geminicephfs/mmsearch-luban-universal/group_7/user_astrachang/model/qwen-next-mlm"
readonly SAVE_CHECKPOINT_DIR="/mnt/geminihzceph1/geminicephfs/mmsearch-luban-universal/group_7/user_astrachang/model/qwen-next-mlm-sft-70w"
readonly TOKENIZER_MODEL="/mnt/geminihzceph1/geminicephfs/mmsearch-luban-universal/group_7/user_astrachang/hf-hub/Qwen/Qwen3-Next-80B-A3B-Instruct"
readonly MLM_PATH="../3rdparty/Megatron-LM"

readonly TP_SIZE=2
readonly PP_SIZE=1
readonly CP_SIZE=1
readonly EP_SIZE=32
readonly MICRO_BATCH_SIZE=1
readonly GLOBAL_BATCH_SIZE=512
readonly TRAIN_ITERS=2737
readonly SEQ_LENGTH=$((24*1024))

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

# torch 启动参数
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

# 并行度
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

# trainer 的 GBS、LR 等
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
"



# 数据与 tokenizer
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
    --tensorboard-dir tb/sft_qwen \
    --tensorboard-log-interval 1 \
    --eval-interval 9999999999 \
    --eval-iters 1 \

    --wandb-project qwen-next-70w \
    --wandb-exp-name qwen-next-sft-tp-${TP_SIZE}-cp-${CP_SIZE} \
    --wandb-save-dir wandb \
"

# 热启动之后去掉这些 flag
FINETUNE_ARGS="
    --finetune \
    --no-load-optim \
    --no-load-rng \
"

export PYTHONPATH="$MLM_PATH:$PWD:$PYTHONPATH"

torchrun $DISTRIBUTED_ARGS tasks/math_rl_v3/sft.py \
    $TRAINER_ARGS \
    $MP_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $FINETUNE_ARGS \
    --distributed-backend nccl \
    --cli-arg-yaml-cfgs gpatch/model_yamls/qwen3-next-80b-a3b.yaml \
    --save $SAVE_CHECKPOINT_DIR \
    --load $LOAD_CHECKPOINT_DIR
