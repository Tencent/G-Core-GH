#!/bin/bash
pip install -U transformers
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
# readonly LOAD_CHECKPOINT_DIR="/mnt/ceph-hz1-csp/mm-base-plt2/user_astrachang/code/gcore-dev/gpt_oss_20b_gcore"
readonly LOAD_CHECKPOINT_DIR="./gpt-oss-20b-bf16-mlm"
readonly SAVE_CHECKPOINT_DIR="$PWD/gpt_oss_20b_sft"
# readonly TOKENIZER_MODEL="/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/openai/gpt-oss-20b/"
readonly TOKENIZER_MODEL="../../../nrwu/hf-hub/lmsys/gpt-oss-20b-bf16"
readonly MLM_PATH="../3rdparty/Megatron-LM"

readonly TP_SIZE=4
readonly PP_SIZE=1
readonly CP_SIZE=2
readonly EP_SIZE=16
readonly MICRO_BATCH_SIZE=1
readonly GLOBAL_BATCH_SIZE=32
readonly TRAIN_ITERS=1024
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
    --attention-backend fused \
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
    --lr 0 \
    --min-lr 0 \
    --lr-warmup-iters 1 \
    --lr-decay-style cosine \
    --optimizer adam \
    --weight-decay 0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --sft \
    --cp-comm-type a2a \
"

# 数据与 tokenizer
DATA_ARGS="
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --dataloader-type external \
    --num-workers 1 \
    --px-data-config-path tasks/gpt_oss/sft_metadata.json \
    --px-shuffle-data \
    --px-shuffle-buffer-size 102400 \
    --px-use-indexed-jsonl-dataset \
    --px-auto-cal-eval-iters \
"

# export WANDB_BASE_URL=
# export WANDB_API_KEY=

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --tensorboard-dir tb/sft_qwen \
    --tensorboard-log-interval 1 \
    --eval-interval 1000 \
    --eval-iters 2 \
"
    # --wandb-project test \
    # --wandb-exp-name qwen2.5-sft-tp-${TP_SIZE}-cp-${CP_SIZE} \
    # --wandb-save-dir wandb \
# 热启动之后去掉这些 flag
FINETUNE_ARGS="
    --finetune \
    --no-load-optim \
    --no-load-rng \
"

export PYTHONPATH="$MLM_PATH:$PWD:$PYTHONPATH"

torchrun $DISTRIBUTED_ARGS tasks/gpt_oss/sft.py \
    $TRAINER_ARGS \
    $MP_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $FINETUNE_ARGS \
    --distributed-backend nccl \
    --cli-arg-yaml-cfgs gpatch/model_yamls/gpt-oss-20b.yaml \
    --save $SAVE_CHECKPOINT_DIR \
    --load $LOAD_CHECKPOINT_DIR
