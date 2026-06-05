#!/bin/bash

ps -ef | grep python | awk  '{print $2}' | xargs -I {} kill -9 {}
sleep 1

set -ex

export PYTHONPATH="$PWD:../../Megatron-LM:$PYTHONPATH"
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
readonly DFS_FOLDER='/mnt/ceph-hz1-csp/mm-base-plt2'
readonly LOAD_CHECKPOINT_DIR="$PWD/deepseek-v2-lite-mlm"
readonly SAVE_CHECKPOINT_DIR="$PWD/deepseek-v2-lite-sft"
readonly TOKENIZER_MODEL="$DFS_FOLDER/nrwu/hf-hub/deepseek-ai/DeepSeek-V2-Lite"

readonly EXP_NAME="deepseek-v2-lite-align"
readonly TB_DIR="tb/deepseek-v2-lite-align"
readonly WANDB_DIR="wandb-save/deepseek-v2-lite-align"

readonly TP_SIZE=1
readonly PP_SIZE=1
readonly EP_SIZE=8
readonly CP_SIZE=1
readonly MICRO_BATCH_SIZE=1
readonly GLOBAL_BATCH_SIZE=8

readonly TRAIN_ITERS=6496
readonly EVAL_ITERS=0
readonly SEQ_LENGTH=2048

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
    --sequence-parallel \
    --context-parallel-size $CP_SIZE \
    --expert-model-parallel-size $EP_SIZE \
    --use-distributed-optimizer \
    --attention-backend fused \
"

readonly MODEL_YAML="gpatch/model_yamls/deepseek-v2-lite.yaml"

LEARNING_RATE=0
TRAINER_ARGS="
    --seq-length $SEQ_LENGTH \
    --use-mcore-models \
    --sequence-parallel \
    --use-flash-attn \
    --disable-bias-linear \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --no-bias-swiglu-fusion \
    --no-check-for-nan-in-loss-and-grad \
    --no-rope-fusion \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --lr ${LEARNING_RATE} \
    --lr-warmup-init ${LEARNING_RATE} \
    --min-lr ${LEARNING_RATE} \
    --lr-decay-style constant \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --train-iters ${TRAIN_ITERS} \
"

    # --data-path /mnt/ceph-hz1-csp/mm-base-plt2/user_jeffhong/workspace/wepsdl-dev/stanford_alpaca/dataset/alpaca_data_sample_1.json \
DATA_ARGS="
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --dataloader-type single \
    --num-workers 1 \
    --data-path /mnt/ceph-hz1-csp/mm-base-plt2/user_jeffhong/workspace/wepsdl-dev/stanford_alpaca/dataset/alpaca_data.json \
    --use-map-dataset \
"

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
    --wandb-exp-name $TS/deepseek-v2-lite-align-tp-${TP_SIZE}-pp-${PP_SIZE}-cp-${CP_SIZE}-ep-${EP_SIZE}-lr-${LEARNING_RATE} \
    --wandb-save-dir wandb \
"

# 热启动之后去掉这些 flag
FINETUNE_ARGS="
    --finetune \
    --no-load-optim \
    --no-load-rng \
"

# RUN_PY='tasks/deepseek/V2-Lite/align_stanford_alpaca/fine_tune.py'
RUN_PY='tools/align_loss/finetune_qwen.py'
# export PX_DEBUG_TRAIN_LOG=1
torchrun $DISTRIBUTED_ARGS $RUN_PY \
    $MP_ARGS \
    $DATA_ARGS \
    --seed 1111 \
    --eod-mask-loss \
    --load-model-provider pretrain_gpt \
    $TRAINER_ARGS \
    $EVAL_AND_OUTPUT_ARGS \
    $FINETUNE_ARGS \
    --distributed-backend nccl \
    --cli-arg-yaml-cfgs $MODEL_YAML \
    --save $SAVE_CHECKPOINT_DIR \
    --load $LOAD_CHECKPOINT_DIR
