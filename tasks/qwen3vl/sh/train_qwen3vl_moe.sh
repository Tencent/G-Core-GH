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


MYWD=$PWD
readonly HF_HUB_DIR="$MYWD/hf-hub/Qwen/Qwen3-VL-30B-A3B-Instruct"

readonly LOAD_CHECKPOINT_DIR=$HF_HUB_DIR
readonly SAVE_CHECKPOINT_DIR="$PWD/ckpt_qwen3vl_30b_a3b_save"
readonly TOKENIZER_MODEL=${HF_HUB_DIR}
readonly MODEL_YAML="gpatch/model_yamls/qwen3vl-30b-a3b-moe.yaml"

# build the meta json file
readonly LMDB_PORT=8312

readonly DATASET_ROOT="${MYWD}/hf-hub/RadGenome/PMC-VQA/gcore-data"
readonly LMDB_PATH="${DATASET_ROOT}/img_file.lmdb"
readonly DATASET_META="/tmp/filter_4k_pmc_vqa_gdataset_v4.json"
python tools/data_convert/build_dataset_v4_meta.py \
    --name "PMC-VQA" \
    --description "the dataset from RadGenome/PMC-VQA" \
    --lmdb_port $LMDB_PORT \
    --output_fullpath $DATASET_META \
    --json_inputs $DATASET_ROOT/filter_4k_qwen3vl/train_2.csv.jsonl \
    --rebuild

DATASET_ARGS="
    --gdatasetv4-train-metadata-file $DATASET_META \
    --px-inputs-pad-to-longest \
"

# readonly DATASET_ROOT="${MYWD}/hf-hub/BUAADreamer/llava-en-zh-300k/gcore-data/zh"
# readonly LMDB_PATH="${DATASET_ROOT}/img_file.lmdb"
# readonly DATASET_META="/tmp/lava-en-zh-300k_dataset_v3.json"
# python tools/data_convert/build_dataset_v3_meta.py \
#     --output_fullpath $DATASET_META \
#     --dataset_dir $DATASET_ROOT \
#     --rebuild

# DATASET_ARGS="
#     --px-data-config-path $DATASET_META \
#     --lmdb-port $LMDB_PORT \
# "

readonly TP_SIZE=2
readonly PP_SIZE=2
readonly EP_SIZE=8
readonly CP_SIZE=1
readonly DP_SIZE=$(($GPUS_PER_NODE*$NNODES/$TP_SIZE/$PP_SIZE/$CP_SIZE))
readonly MICRO_BATCH_SIZE=1
readonly GLOBAL_BATCH_SIZE=256

readonly TRAIN_ITERS=595
readonly LR_WARMUP_ITERS=10
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

if [ $PP_SIZE -lt 2 ]; then
    PP_FIRST_LAST_LAYERS=""
else
    readonly NUM_LAYERS=$(yq  '.num_layers' $MODEL_YAML)
    FIRST_LAST_LAYER=$((NUM_LAYERS - (NUM_LAYERS + PP_SIZE - 1) / PP_SIZE * (PP_SIZE - 2)))
    if [ $FIRST_LAST_LAYER -le 1 ]; then
        echo "Error: FIRST_LAST_LAYER must be greater than 1"
        exit 1
    fi
    FIRST_LAYER=$((FIRST_LAST_LAYER / 2))
    LAST_LAYER=$(((FIRST_LAST_LAYER + 1) / 2))
    echo "--decoder-first-pipeline-num-layers: $FIRST_LAYER"
    echo "--decoder-last-pipeline-num-layers: $LAST_LAYER"

    PP_FIRST_LAST_LAYERS="
        --decoder-first-pipeline-num-layers $FIRST_LAYER \
        --decoder-last-pipeline-num-layers $LAST_LAYER \
    "
fi

# 并行度
MP_ARGS="
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE 
    --expert-model-parallel-size $EP_SIZE \
    --expert-tensor-parallel-size 1 \
    --sequence-parallel \
    --context-parallel-size $CP_SIZE \
    --use-distributed-optimizer \
    --attention-backend auto \
    $PP_FIRST_LAST_LAYERS \
"

# trainer 的 GBS、LR 等
TRAINER_ARGS="
    --lr 5e-7 \
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
    --seq-length 4096 \
    --decoder-seq-length 4096 \
    --use-rotary-position-embeddings \
    --rotary-percent 1.0 \
    --rotary-seq-len-interpolation-factor 1 \
    --seed 42 \
    --ckpt-format torch_dist \
    --no-rope-fusion \
    --no-gradient-accumulation-fusion \
    --processor-path ${TOKENIZER_MODEL} \
    --mask-history \
    --try-load-from-hf \
    --hf-model-path ${HF_HUB_DIR} \
    --attention-softmax-in-fp32 \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --recompute-granularity full \
    --export-to-hf \
    --mbridge-distributed-filesystem \
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
    --px-shuffle-buffer-size 102400 \
    --px-smart-padding-buffer-size 2048 \
    --px-pad-to-multiple-of 128 \
    --moe-pad-with-random-token \
"

export WANDB_API_KEY=
export WANDB_BASE_URL=
OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 100 \
    --eval-interval 1000 \
    --eval-iters $EVAL_ITERS \
    --tensorboard-dir tb/qwen3vl \
    --tensorboard-log-interval 1 \
    --wandb-project qwen3vl-base \
    --wandb-exp-name sft/dp-${DP_SIZE}/tp${TP_SIZE}-pp${PP_SIZE}-cp${CP_SIZE}-mbs${MICRO_BATCH_SIZE}-gbs${GLOBAL_BATCH_SIZE} \
    --wandb-save-dir wandb \
"

# 热启动之后去掉这些 flag
FINETUNE_ARGS="
    --finetune \
    --no-load-optim \
    --no-load-rng \
"

pkill -f -9 lmdb_read_svr.py
nohup python megatron_datasets/tools/lmdb_read_svr.py \
    --lmdb-path $LMDB_PATH \
    --lmdb-map-size 500 \
    --lmdb-port $LMDB_PORT > svr.log 2>&1 &

readonly MLM_PATH="../3rdparty/Megatron-LM:../Megatron-LM"
readonly MBRIDGE_PATH="../mbridge"
export PYTHONPATH="$MBRIDGE_PATH:$MLM_PATH:$PYTHONPATH"

PYTHONPATH="${PWD}:$PYTHONPATH" torchrun $DISTRIBUTED_ARGS \
    tasks/qwen3vl/train_qwen3vl.py \
    $TRAINER_ARGS \
    $MP_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $FINETUNE_ARGS \
    --distributed-backend nccl \
    --cli-arg-yaml-cfgs $MODEL_YAML \
    --save $SAVE_CHECKPOINT_DIR \
    --load $LOAD_CHECKPOINT_DIR 

pkill -f -9 lmdb_read_svr.py
