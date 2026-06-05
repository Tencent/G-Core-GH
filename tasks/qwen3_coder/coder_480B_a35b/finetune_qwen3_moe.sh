#!/bin/bash
ps -ef | grep python | awk  '{print $2}' | xargs -I {} kill -9 {}
sleep 1

# 新镜像为什么把 python3 变成了系统默认自带的？ 暂时这样软链回去吧
rm /bin/python3
ln -s /root/conda/bin/python /bin/python3

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

MLM_PATH="../3rdparty/Megatron-LM/"
export PYTHONPATH="$PWD:$MLM_PATH:$PYTHONPATH"
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

export WANDB_BASE_URL=
export WANDB_API_KEY=

readonly USERNAME="qwen3_coder"

DFS_PATH="/mnt/ceph-sh2/models"

# qwen3 moe 30B-a3b
readonly MODEL_YAML="gpatch/model_yamls/qwen3-coder-480b-a35b-instruct.yaml"
readonly LOAD_CHECKPOINT_DIR="$PWD/qwen3_coder_480b_a35b_moe_mlm"
readonly SAVE_CHECKPOINT_DIR="$PWD/qwen3_coder_480b_a35b_moe_mlm_save"
readonly TOKENIZER_MODEL="$DFS_PATH/Qwen3-Coder-480B-A35B-Instruct/"
readonly EXP_NAME="qwen3_coder_480b_32k"
readonly TB_DIR="tb/qwen3_coder_480b_32k"
readonly WANDB_DIR="wandb-save/qwen3_coder_480b_32k"

# 注意，建议设置自己的 system_prompt, 并且在下面 args 打开开关
#     --px-apply-chat-template \
#     --px-system-prompt "$system_prompt" \
# 如果无 system_prompt, 则只打开 --px-apply-chat-template 即可


readonly TP_SIZE=2
readonly PP_SIZE=8
readonly EP_SIZE=16
readonly CP_SIZE=4
readonly DP_SIZE=$(($GPUS_PER_NODE*$NNODES/$TP_SIZE/$PP_SIZE/$CP_SIZE))
readonly MICRO_BATCH_SIZE=1
readonly GLOBAL_BATCH_SIZE=256


readonly TRAIN_ITERS=1000
readonly EVAL_ITERS=0
readonly SEQ_LEN=$((1024*32))


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
    --decoder-first-pipeline-num-layers 7 \
    --decoder-last-pipeline-num-layers 7 \
    --context-parallel-size $CP_SIZE \
    --expert-model-parallel-size $EP_SIZE \
    --expert-tensor-parallel-size 1 \
    --sequence-parallel \
    --use-distributed-optimizer \
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
    --lr-warmup-iters 200 \
    --lr-decay-style cosine \
    --optimizer adam \
    --weight-decay 0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --attention-backend fused \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --recompute-granularity full \
"

DATA_ARGS="
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --dataloader-type external \
    --num-workers 1 \
    --px-data-config-path tasks/math_rl_v3/qwen/sft_data_config.json \
    --px-shuffle-data \
    --px-shuffle-buffer-size 10000 \
    --px-use-indexed-jsonl-dataset \
    --px-auto-cal-eval-iters \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --tensorboard-dir $TB_DIR \
    --tensorboard-log-interval 1 \
    --eval-interval 1 \
    --eval-iters $EVAL_ITERS \
    --no-save-rng \
    --no-save-optim \
    --wandb-project $USERNAME \
    --wandb-exp-name $EXP_NAME \
    --wandb-save-dir wandb-save \
    --log-throughput \
"

EXPERT_ARGS="
    --moe-pad-with-random-token \
    --overlap-grad-reduce \
    --overlap-param-gather \
"

# 热启动之后去掉这些 flag
FINETUNE_ARGS="
    --finetune \
    --no-load-optim \
    --no-load-rng \
"


RUN_PY="tasks/math_rl_v3/sft.py"
torchrun $DISTRIBUTED_ARGS $RUN_PY \
    $TRAINER_ARGS \
    $MP_ARGS \
    $EXPERT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $FINETUNE_ARGS \
    --cli-arg-yaml-cfgs $MODEL_YAML \
    --distributed-backend nccl \
    --px-apply-chat-template \
    --save $SAVE_CHECKPOINT_DIR \
    --load $LOAD_CHECKPOINT_DIR
