#!/bin/bash

readonly MPI_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly MPI_SIZE="${OMPI_COMM_WORLD_SIZE:-1}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME="bond1"
export GLOO_SOCKET_IFNAME="bond1"
export NCCL_DEBUG=WARN
export RAY_DEDUP_LOGS=0

DEVICE_TYPE='cuda'

readonly NODE_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly NNODES="${OMPI_COMM_WORLD_SIZE:-1}"
readonly GPUS_PER_NODE=${GPUS_PER_NODE:-8}
readonly WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

readonly TOPO_CONFIG_FOLDER=$1

MYWD=$PWD
# 理论上只需要修改：HF_HUB_DIR 与 MODEL_YAML
INFER_ENGINE=sglang # 只支持 vllm / sglang
# readonly HF_HUB_DIR="$MYWD/hf-hub/Qwen/Qwen2.5-VL-3B-Instruct"
# readonly MODEL_YAML="gpatch/model_yamls/qwen2.5-vl-3b.yaml"

readonly HF_HUB_DIR="$MYWD/hf-hub/Qwen/Qwen3-VL-30B-A3B-Instruct"
readonly MODEL_YAML="gpatch/model_yamls/qwen3vl-30b-a3b-moe.yaml"

readonly TOKENIZER_MODEL=$HF_HUB_DIR
readonly GPU_UTIL=0.6
readonly MAX_REQ=128

SAMPLER_NNODES=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn sampler-nnodes`
SAMPLER_MASTER_ADDR=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn sampler-master-addr`
SAMPLER_SVR_IPS=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn sampler-svr-ips`
SAMPLER_SVR_PORTS=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn sampler-svr-ports`
SAMPLER_DIST_INIT_ADDRS=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn sampler-dist-init-addrs`
SAMPLER_TP_SIZE=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn sampler-tp-size`
SAMPLER_PP_SIZE=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn sampler-pp-size`

export MASTER_ADDR="${SAMPLER_MASTER_ADDR:-localhost}"
readonly MASTER_PORT=65204

readonly LOAD_CHECKPOINT_DIR=$HF_HUB_DIR
readonly SAVE_CHECKPOINT_DIR="none"
readonly REF_LOAD_CHECKPOINT_DIR=$LOAD_CHECKPOINT_DIR

readonly CRITIC_DP_SIZE=$(($GPUS_PER_NODE*$CRITIC_NNODES/$CRITIC_TP_SIZE/$CRITIC_PP_SIZE))
readonly SAMPLER_DP_SIZE=$(($GPUS_PER_NODE*$SAMPLER_NNODES/$SAMPLER_TP_SIZE/$SAMPLER_PP_SIZE))
readonly ACTOR_DP_SIZE=$(($GPUS_PER_NODE*$ACTOR_NNODES/$ACTOR_TP_SIZE/$ACTOR_PP_SIZE/$ACTOR_CP_SIZE))

readonly TP_SIZE=$SAMPLER_TP_SIZE
readonly PP_SIZE=$SAMPLER_PP_SIZE
readonly EP_SIZE=1
readonly CP_SIZE=1
readonly DP_SIZE=$SAMPLER_DP_SIZE


readonly SEQ_LENGTH=4096

echo "INFO
MPI_RANK $MPI_RANK
NODE_RANK $NODE_RANK
NNODES $NNODES
TP_SIZE $TP_SIZE
PP_SIZE $PP_SIZE
EP_SIZE $EP_SIZE
CP_SIZE $CP_SIZE
DP_SIZE $DP_SIZE
SAMPLER_SVR_IPS $SAMPLER_SVR_IPS
SAMPLER_SVR_PORTS $SAMPLER_SVR_PORTS
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
    --use-tp-pp-dp-mapping \
"

TRAINER_ARGS="
    --micro-batch-size 1 \
    --global-batch-size 15360 \
    --seq-length $SEQ_LENGTH \
    --decoder-seq-length $SEQ_LENGTH \
"

DATA_ARGS="
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval -1 \
    --eval-interval 1 \
"

readonly DIST_TIMEOUT_MIN=300

RL_ARGS="
    --distributed-timeout-minutes $DIST_TIMEOUT_MIN \
    --infer-engine-impl $INFER_ENGINE \
    --ppo-sampler-ips $SAMPLER_SVR_IPS \
    --ppo-sampler-ports $SAMPLER_SVR_PORTS \
    --sampler-dist-init-addrs $SAMPLER_DIST_INIT_ADDRS \
    --ppo-sampler-tensor-model-parallel-size $SAMPLER_TP_SIZE \
    --ppo-sampler-pipeline-model-parallel-size $SAMPLER_PP_SIZE \
    --ppo-sampler-data-parallel-size $SAMPLER_DP_SIZE \
    --sampler-gpu-memory-utilization $GPU_UTIL \
    --max-running-requests $MAX_REQ \
"

if [ "$INFER_ENGINE" = "vllm" ]; then
    # sampler的环境变量应该尽可能在ray start前设置
    export VLLM_HOST_IP=`ifconfig bond1 | grep inet | grep broadcast | awk -F' ' '{print $2}'`
    echo "VLLM_HOST_IP: $VLLM_HOST_IP"
    python tools/auto_place.py --fn init_ray --config-folder $TOPO_CONFIG_FOLDER
fi

RUN_PY='./tools/eval/eval_server.py'
readonly MLM_PATH=../Megatron-LM
export PYTHONPATH="$MLM_PATH:$PYTHONPATH"

torchrun $DISTRIBUTED_ARGS $RUN_PY \
    $TRAINER_ARGS \
    $MP_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $RL_ARGS \
    --cli-arg-yaml-cfgs $MODEL_YAML \
    --distributed-backend nccl \
    --save $SAVE_CHECKPOINT_DIR \
    --load $LOAD_CHECKPOINT_DIR \
    --load-ref $REF_LOAD_CHECKPOINT_DIR
