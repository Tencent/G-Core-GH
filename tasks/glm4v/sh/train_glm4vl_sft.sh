#!/bin/bash
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
export IMAGE_TOKEN_ID=151363

MYWD=$PWD
readonly HF_HUB_DIR="$MYWD/hf-hub/zai-org/GLM-4.5V"

# can be load from hf
readonly LOAD_CHECKPOINT_DIR=${HF_HUB_DIR}
readonly SAVE_CHECKPOINT_DIR="./ckpt_glm4p5vl_save_sft"

MYWD=$PWD
readonly TOKENIZER_MODEL=${HF_HUB_DIR}
readonly MODEL_YAML="gpatch/model_yamls/glm4.5v.yaml"

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
    --json_inputs $DATASET_ROOT/filter_4k/train_2.csv.jsonl \
    --rebuild

DATASET_ARGS="
    --gdatasetv4-train-metadata-file $DATASET_META \
    --px-inputs-pad-to-longest \
"

readonly TP_SIZE=4
readonly PP_SIZE=1
readonly EP_SIZE=8
readonly ETP_SIZE=1
readonly CP_SIZE=1
readonly DP_SIZE=$(($GPUS_PER_NODE*$NNODES/$TP_SIZE/$PP_SIZE/$CP_SIZE))


readonly MICRO_BATCH_SIZE=4
readonly GLOBAL_BATCH_SIZE=256

readonly TRAIN_ITERS=500
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
GRADIENT_ACCUMULATE_STEP $GRADIENT_ACCUMULATE_STEP
GLOBAL_BATCH_SIZE $GLOBAL_BATCH_SIZE
HCCL_BUFFSIZE $HCCL_BUFFSIZE
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
    --expert-model-parallel-size $EP_SIZE \
    --expert-tensor-parallel-size $ETP_SIZE \
    --sequence-parallel \
    --context-parallel-size $CP_SIZE \
    --use-distributed-optimizer \
    --attention-backend flash \
    --decoder-first-pipeline-num-layers 11 \
    --decoder-last-pipeline-num-layers 11 \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --recompute-granularity full \
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
    --image-token-id $IMAGE_TOKEN_ID \
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
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 100 \
    --eval-interval 1000 \
    --eval-iters $EVAL_ITERS \
    --tensorboard-dir tb/sft-glm4.5vl \
    --tensorboard-log-interval 1 \
"

<< 'COMMENT'
--wandb-project glm4vl-base \
--wandb-exp-name sft/tp-${TP_SIZE}-cp-${CP_SIZE} \
--wandb-save-dir wandb \
COMMENT


SFT_ARGS=" --mm-freeze-vision-encoder "
    


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


readonly MLM_PATH=../Megatron-LM
readonly MBRIDGE_PATH=../mbridge
export PYTHONPATH="$MLM_PATH:${MBRIDGE_PATH}:${MYWD}:${PYTHONPATH}"

run_cmd="torchrun $DISTRIBUTED_ARGS \
    tasks/glm4v/train_glm4vl.py \
    $TRAINER_ARGS \
    $MP_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $FINETUNE_ARGS \
    $SFT_ARGS \
    --distributed-backend nccl \
    --cli-arg-yaml-cfgs $MODEL_YAML \
    --save $SAVE_CHECKPOINT_DIR \
    --load $LOAD_CHECKPOINT_DIR \
"

${run_cmd}

pkill -f -9 lmdb_read_svr.py
