# coding=utf-8
# copyright (c) 2025 tencent inc. all rights reserved.
# astrachang@tencent.com, yeazhao@tencent.com

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDNN_PATH=/usr/lib64

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_DATASETS_OFFLINE=1

readonly GPUS_PER_NODE=8
readonly NODE_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly NNODES="${OMPI_COMM_WORLD_SIZE:-1}"
readonly WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
readonly MASTER_PORT=25365
export MASTER_ADDR="${_MASTER_ADDR:-localhost}"

# /mnt/gemininjceph2/geminicephfs/mm-base-plt2/nrwu/hf-hub/Qwen/Qwen3-30B-A3B/
readonly MODEL_YAML="gpatch/model_yamls/gpt-oss-20b.yaml"
readonly DFS_FOLDER="/mnt/ceph-hz1-csp/mm-base-plt2"
# readonly LOAD_CHECKPOINT_DIR="/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_7/user_astrachang/code/gcore-dev/./gpt-oss-120b-mlm-final"
readonly LOAD_CHECKPOINT_DIR="./20b_dev"
readonly SAVE_CHECKPOINT_DIR="$PWD/gpt_oss_120b_mlm"
readonly TOKENIZER_MODEL="/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_7/nrwu/hf-hub/lmsys/gpt-oss-20b-bf16"
readonly TB_DIR="tb/gpt_oss_20b_mlm"

readonly SEQ_LEN=512
readonly TP_SIZE=8
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
    --sequence-parallel \
"


TRAINER_ARGS="
    --seq-length $SEQ_LEN \
    --seed 1111 \
    --eod-mask-loss \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
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
    --attention-backend fused \
"

DATA_ARGS="
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
"

EXPERT_ARGS="
    --moe-grouped-gemm \
    --moe-token-dispatcher-type allgather \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1 \
    --eval-interval 1 \
    --eval-iters 1 \
    --load-model-provider tasks.gpt_oss.sft \
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
# 定义prompts数组
PROMPTS=(
    "what's the result of 1+1"
    "write a poem about openai"
    "do you know tomorin?"
    "what's your name?"
)

GEN_ARGS="
    --max-new-tokens 128 \
"

# readonly MLM_PATH="/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_7/user_oriontian/gpt-oss/Megatron-MoE-EA"
readonly MLM_PATH="../3rdparty/Megatron-LM"

PYTHONPATH="$PWD:$MLM_PATH:$PYTHONPATH" \
torchrun $DISTRIBUTED_ARGS tasks/gpt_oss/mlm_infer.py \
    $TRAINER_ARGS \
    $MP_ARGS \
    $EXPERT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $GEN_ARGS \
    --cli-arg-yaml-cfgs $MODEL_YAML \
    --distributed-backend nccl \
    --save $SAVE_CHECKPOINT_DIR \
    --load $LOAD_CHECKPOINT_DIR \
    --prompts "${PROMPTS[@]}" \
    --inference-max-requests 10
