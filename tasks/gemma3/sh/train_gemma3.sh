#!/bin/bash
# $1: train type: sft/dpo

readonly TRAIN_TYPE=${1:-sft}

ps -ef | grep python | awk  '{print $2}' | xargs -I {} kill -9 {}
sleep 1

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

readonly LOAD_CHECKPOINT_DIR="$PWD/ckpt_gemma3_$TRAIN_TYPE"
readonly SAVE_CHECKPOINT_DIR="$PWD/ckpt_gemma3_save_$TRAIN_TYPE"

MYWD=$PWD
readonly TOKENIZER_MODEL="$MYWD/hf-hub/google/gemma-3-4b-it"
readonly MODEL_YAML="gpatch/model_yamls/gemma3-4b.yaml"

# build the meta json file
readonly LMDB_PORT=8312
if [ "$TRAIN_TYPE" = "sft" ]; then
    readonly DATASET_ROOT="${MYWD}/hf-hub/BUAADreamer/llava-en-zh-300k/gcore-data/zh"
    readonly LMDB_PATH="${DATASET_ROOT}/img_file.lmdb"
    readonly DATASET_META="/tmp/lava-en-zh-300k_dataset_v3.json"
    python tools/data_convert/build_dataset_v3_meta.py \
        --output_fullpath $DATASET_META \
        --dataset_dir $DATASET_ROOT \
        --rebuild

    DATASET_ARGS="
        --px-data-config-path $DATASET_META \
        --lmdb-port $LMDB_PORT \
    "
elif [ "$TRAIN_TYPE" = "dpo" ]; then
    readonly DATASET_ROOT="${MYWD}/hf-hub/llamafactory/RLHF-V/gcore-data"
    readonly LMDB_PATH="${DATASET_ROOT}/img_file.lmdb"
    readonly DATASET_META="/tmp/RLHF-V_dataset_v3.json"
    python tools/data_convert/build_dataset_v3_meta.py \
        --output_fullpath $DATASET_META \
        --dataset_dir $DATASET_ROOT \
        --rebuild

    # DPO not support --px-inputs-pad-to-longest now
    DATASET_ARGS="
        --px-data-config-path $DATASET_META \
        --lmdb-port $LMDB_PORT \
    "
else
    echo "not support this train type:${TRAIN_TYPE}"
    exit 0
fi

readonly TP_SIZE=2
readonly PP_SIZE=2
readonly EP_SIZE=1
readonly CP_SIZE=2
readonly DP_SIZE=$(($GPUS_PER_NODE*$NNODES/$TP_SIZE/$PP_SIZE/$CP_SIZE))
if [ "$TRAIN_TYPE" = "dpo" ]; then
    readonly MICRO_BATCH_SIZE=2
else
    readonly MICRO_BATCH_SIZE=1
fi
readonly GLOBAL_BATCH_SIZE=32

readonly TRAIN_ITERS=20000
readonly LR_WARMUP_ITERS=0
readonly EVAL_ITERS=10
LR_DECAY_ITERS=$(( ${TRAIN_ITERS} - ${LR_WARMUP_ITERS}))

echo "INFO
NODE_RANK $NODE_RANK
NNODES $NNODES
TP_SIZE $TP_SIZE
PP_SIZE $PP_SIZE
EP_SIZE $EP_SIZE
CP_SIZE $CP_SIZE
DP_SIZE $DP_SIZE
MICRO_BATCH_SIZE $MICRO_BATCH_SIZE
GLOBAL_BATCH_SIZE $GLOBAL_BATCH_SIZE
"

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
    --use-distributed-optimizer \
    --attention-backend flash \
"

TRAINER_ARGS="
    --lr 0 \
    --min-lr 0 \
    --lr-decay-style constant \
    --weight-decay 0.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-08 \
    --clip-grad 1.0 \
    --lr-decay-iters ${LR_DECAY_ITERS} \
    --lr-warmup-iters ${LR_WARMUP_ITERS} \
    --train-iters ${TRAIN_ITERS} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --attention-softmax-in-fp32 \
    --transformer-impl transformer_engine \
    --rotary-percent 1.0 \
    --bf16 \
    --seq-length 256 \
    --decoder-seq-length 2048 \
    --img-h 896 \
    --img-w 896 \
    --patch-dim 14 \
    --seed 42 \
    --ckpt-format torch_dist \
    --processor-path ${TOKENIZER_MODEL} \
    --qk-layernorm \
    --mask-history \
    --apply-layernorm-1p \
"

DATA_ARGS="
    $DATASET_ARGS \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --dataloader-type external \
    --num-workers 8 \
    --timing-log-level 1 \
    --px-reset-dataloader-at-start-of-eval \
    --px-dataloader-prefetch-factor 32 \
    --eod-mask-loss \
"

export WANDB_API_KEY=
export WANDB_BASE_URL=

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-iters $EVAL_ITERS \
    --tensorboard-dir tb/$TRAIN_TYPE-gemma3 \
    --tensorboard-log-interval 1 \
    --wandb-project gemma3-base \
    --wandb-exp-name test/$TRAIN_TYPE/dp-${DP_SIZE}/tp${TP_SIZE}-pp${PP_SIZE}-cp${CP_SIZE}-mbs${MICRO_BATCH_SIZE}-gbs${GLOBAL_BATCH_SIZE} \
    --wandb-save-dir wandb \
    --padded-vocab-size 262272 \
"

if [ $TRAIN_TYPE == "sft" ]; then
    SFT_ARGS=""
    DPO_ARGS=""
elif [ $TRAIN_TYPE == "dpo" ]; then
    SFT_ARGS=""
    DPO_ARGS="
        --dpo \
        --dpo-beta 0.1 \
        --dpo-label-smoothing 0. \
        --dpo-ftx-gamma 0. \
        --dpo-reward-models-cnt 0 \
        --dpo-margin-keys rel faithful formater complete \
        --dpo-policy-ref-model-cnt 2 \
    "
else
    echo "not support this train type:${TRAIN_TYPE}"
    exit 0
fi

# 热启动之后去掉这些 flag
FINETUNE_ARGS="
    --finetune \
    --no-load-optim \
    --no-load-rng \
"

nohup python megatron_datasets/tools/lmdb_read_svr.py \
    --lmdb-path $LMDB_PATH \
    --lmdb-map-size 500 \
    --lmdb-port $LMDB_PORT > svr.log 2>&1 &

readonly MLM_PATH=../Megatron-LM
export PYTHONPATH="$MLM_PATH:$PYTHONPATH"

PYTHONPATH="${PWD}:$PYTHONPATH" torchrun $DISTRIBUTED_ARGS \
    tasks/gemma3/train_gemma3.py \
    $TRAINER_ARGS \
    $MP_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $FINETUNE_ARGS \
    $SFT_ARGS \
    $LORA_ARGS \
    $DPO_ARGS \
    --distributed-backend nccl \
    --cli-arg-yaml-cfgs $MODEL_YAML \
    --save $SAVE_CHECKPOINT_DIR \
    --load $LOAD_CHECKPOINT_DIR 

pkill -f -9 lmdb_read_svr.py
