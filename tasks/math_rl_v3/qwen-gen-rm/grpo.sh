#!/bin/bash

readonly MPI_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly MPI_SIZE="${OMPI_COMM_WORLD_SIZE:-1}"

export PYTHONPATH="$PWD:/root/Megatron-LM:$PYTHONPATH"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME="bond1"
export GLOO_SOCKET_IFNAME="bond1"
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=WARN
export RAY_DEDUP_LOGS=0

# 设置 max_split_size_mb 避免切分过大的显存 block 导致显存碎片
# 默认值是 -1，表示任意大小的显存 block 都可以被切分
# 如果显存碎片问题严重到导致 CUDA OOM，可尝试设为 256 (empirical)
# max_split_size_mb 越小，越能缓解显存碎片问题，但性能可能会下降，建议不要设置太小，并且观察吞吐表现
# see https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:-1
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

readonly DIST_TIMEOUT_MIN=300

readonly NODE_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly NNODES="${OMPI_COMM_WORLD_SIZE:-1}"
readonly GPUS_PER_NODE=${GPUS_PER_NODE:-8}
readonly WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

readonly PLACE_CFG_FOLDER=$1
readonly PPO_ROLE=$2

SAMPLER_NNODES=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn sampler-nnodes`
SAMPLER_MASTER_ADDR=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn sampler-master-addr`
SAMPLER_SVR_IPS=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn sampler-svr-ips`
SAMPLER_SVR_PORTS=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn sampler-svr-ports`
SAMPLER_DIST_INIT_ADDRS=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn sampler-dist-init-addrs`
SAMPLER_TP_SIZE=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn sampler-tp-size`
SAMPLER_PP_SIZE=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn sampler-pp-size`

GEN_RM_NNODES=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn gen-rm-nnodes`
GEN_RM_MASTER_ADDR=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn gen-rm-master-addr`
GEN_RM_SVR_IPS=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn gen-rm-svr-ips`
GEN_RM_SVR_PORTS=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn gen-rm-svr-ports`
GEN_RM_DIST_INIT_ADDRS=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn gen-rm-dist-init-addrs`
GEN_RM_TP_SIZE=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn gen-rm-tp-size`
GEN_RM_PP_SIZE=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn gen-rm-pp-size`

ACTOR_NNODES=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn actor-nnodes`
ACTOR_MASTER_ADDR=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn actor-master-addr`
ACTOR_SVR_IPS=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn actor-svr-ips`
ACTOR_SVR_PORTS=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn actor-svr-ports`
ACTOR_NODE_IPS=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn actor-node-ips`
ACTOR_TP_SIZE=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn actor-tp-size`
ACTOR_PP_SIZE=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn actor-pp-size`
ACTOR_CP_SIZE=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn actor-cp-size`

if [ "$PPO_ROLE" = "actor" ]; then
    export MASTER_ADDR="${ACTOR_MASTER_ADDR:-localhost}"
    readonly MASTER_PORT=65531
elif [ "$PPO_ROLE" = "gen-rm" ]; then
    export MASTER_ADDR="${GEN_RM_MASTER_ADDR:-localhost}"
    readonly MASTER_PORT=65535
elif [ "$PPO_ROLE" = "sampler" ]; then
    export MASTER_ADDR="${SAMPLER_MASTER_ADDR:-localhost}"
    readonly MASTER_PORT=65534
else
    echo "$PPO_ROLE no support"
    exit 0
fi

readonly WANDB_PROJECT="test"
readonly WANDB_EXP_NAME="qwen2.5-grpo-gdatasetv4"
export WANDB_BASE_URL=
export WANDB_API_KEY=

readonly ACTOR_TOKENIZER_MODEL="${PWD}/hf-hub/Qwen/Qwen2.5-Math-1.5B/"
readonly RM_TOKENIZER_MODELS="${PWD}/hf-hub/Qwen/Qwen2.5-Math-72B-Instruct/"
readonly data_path="tasks/math_rl_v3/qwen/rl_metadata.json"
readonly eval_data_path="tasks/math_rl_v3/qwen/rl_eval_metadata.json"

if [ "$PPO_ROLE" = "actor" ]; then
    readonly MODEL_YAML="gpatch/model_yamls/qwen2.5-math-1.5b.yaml"
    readonly LOAD_CHECKPOINT_DIR="${PWD}/qwen_2_5_1_5b"
    readonly REF_LOAD_CHECKPOINT_DIR="${PWD}/qwen_2_5_1_5b_ref"
    readonly SAVE_CHECKPOINT_DIR="${PWD}/qwen_2_5_1_5b_save"
    readonly TOKENIZER_MODEL=$ACTOR_TOKENIZER_MODEL
    readonly TB_DIR="tb/grpo-actor"
    readonly WANDB_DIR="wandb_local/grpo-actor"
    readonly LR=5e-7
elif [ "$PPO_ROLE" = "gen-rm" ]; then
    readonly MODEL_YAML="gpatch/model_yamls/qwen2.5-math-72b-instruct.yaml"
    readonly LOAD_CHECKPOINT_DIR="${PWD}/hf-hub/Qwen/Qwen2.5-Math-72B-Instruct"
    readonly REF_LOAD_CHECKPOINT_DIR="${PWD}/hf-hub/Qwen/Qwen2.5-Math-72B-Instruct ${PWD}/hf-hub/Qwen/Qwen2.5-Math-72B-Instruct"
    readonly SAVE_CHECKPOINT_DIR="none"
    readonly TOKENIZER_MODEL=$RM_TOKENIZER_MODELS
    readonly TB_DIR="tb/grpo-gen-rm"
    readonly WANDB_DIR="wandb_local/grpo-gen-rm"
    readonly LR=5e-7
elif [ "$PPO_ROLE" = "sampler" ]; then
    readonly MODEL_YAML="gpatch/model_yamls/qwen2.5-math-1.5b.yaml"
    # LOAD_CHECKPOINT_DIR 填入 hf ckpt 
    readonly LOAD_CHECKPOINT_DIR="${PWD}/hf-hub/Qwen/Qwen2.5-Math-1.5B/"
    readonly REF_LOAD_CHECKPOINT_DIR=$LOAD_CHECKPOINT_DIR
    readonly SAVE_CHECKPOINT_DIR="none"
    readonly TOKENIZER_MODEL=$ACTOR_TOKENIZER_MODEL
    readonly TB_DIR="tb/dqa-ppo-sampler"
    readonly WANDB_DIR="wandb_local/grpo-sampler"
    readonly LR=5e-7
else
    echo "$PPO_ROLE no support"
    exit 0
fi

# 确定 topo
readonly GEN_RM_DP_SIZE=$(($GPUS_PER_NODE*$GEN_RM_NNODES/$GEN_RM_TP_SIZE/$GEN_RM_PP_SIZE))
readonly SAMPLER_DP_SIZE=$(($GPUS_PER_NODE*$SAMPLER_NNODES/$SAMPLER_TP_SIZE/$SAMPLER_PP_SIZE))
readonly SAMPLER_MP_SIZE=$(($SAMPLER_TP_SIZE*$SAMPLER_PP_SIZE))
readonly ACTOR_DP_SIZE=$(($GPUS_PER_NODE*$ACTOR_NNODES/$ACTOR_TP_SIZE/$ACTOR_PP_SIZE/$ACTOR_CP_SIZE))

if [ "$PPO_ROLE" = "actor" ]; then
    readonly TP_SIZE=$ACTOR_TP_SIZE
    readonly PP_SIZE=$ACTOR_PP_SIZE
    readonly EP_SIZE=1
    readonly CP_SIZE=$ACTOR_CP_SIZE
    readonly DP_SIZE=$ACTOR_DP_SIZE
elif [ "$PPO_ROLE" = "gen-rm" ]; then
    readonly TP_SIZE=$GEN_RM_TP_SIZE
    readonly PP_SIZE=$GEN_RM_PP_SIZE
    readonly EP_SIZE=1
    readonly CP_SIZE=1
    readonly DP_SIZE=$GEN_RM_DP_SIZE
elif [ "$PPO_ROLE" = "sampler" ]; then
    readonly TP_SIZE=$SAMPLER_TP_SIZE
    readonly PP_SIZE=$SAMPLER_PP_SIZE
    readonly EP_SIZE=1
    readonly CP_SIZE=1
    readonly DP_SIZE=$SAMPLER_DP_SIZE
fi

readonly ROLLOUT_GLOBAL_BATCH_SIZE=8
readonly ROLLOUT_MICRO_BATCH_SIZE=1
readonly MICRO_BATCH_SIZE=1
readonly SHUFFLE_BUFFER_SIZE=$((256*$ROLLOUT_GLOBAL_BATCH_SIZE))
if [ "$PPO_ROLE" = "actor" ]; then
    readonly GLOBAL_BATCH_SIZE=8
else
    # sampler 和 rm 不需要训练，没有真正的 GBS，设置成 dp size，方便 scaling。
    readonly GLOBAL_BATCH_SIZE=$DP_SIZE
fi

readonly PPO_LOGPS_FWD_MICRO_BATCH_SIZE=1
readonly TRAIN_ITERS=-1
readonly EVAL_ITERS=0
readonly SEQ_LENGTH=$((4*1024))
readonly RESP_SEQ_LENGTH=$((1024))
readonly MAX_SAMPLING_RETRIES=8

echo "INFO
MPI_RANK $MPI_RANK
PPO_ROLE $PPO_ROLE
NODE_RANK $NODE_RANK
NNODES $NNODES
TP_SIZE $TP_SIZE
PP_SIZE $PP_SIZE
EP_SIZE $EP_SIZE
CP_SIZE $CP_SIZE
DP_SIZE $DP_SIZE
GEN_RM_DP_SIZE $GEN_RM_DP_SIZE
MICRO_BATCH_SIZE $MICRO_BATCH_SIZE
GRADIENT_ACCUMULATE_STEP $GRADIENT_ACCUMULATE_STEP
GLOBAL_BATCH_SIZE $GLOBAL_BATCH_SIZE
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

MP_ARGS="
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --sequence-parallel \
    --context-parallel-size $CP_SIZE \
    --use-distributed-optimizer \
"

if [ "$PPO_ROLE" = "sampler" ] || [ "$PPO_ROLE" = "gen-rm" ]; then
    MP_ARGS="$MP_ARGS
        --use-tp-pp-dp-mapping \
    "
fi

TRAINING_ARGS="
    --use-mcore-models \
    --seq-length $SEQ_LENGTH \
    --seed 1111 \
    --no-check-for-nan-in-loss-and-grad \
    --eod-mask-loss \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-iters $TRAIN_ITERS \
    --lr $LR \
    --min-lr 0. \
    --lr-warmup-iters 20 \
    --lr-decay-style cosine \
    --optimizer adam \
    --weight-decay 0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --attention-backend fused \
    --use-flash-attn \
"

DATA_ARGS="
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --actor-tokenizer-model ${ACTOR_TOKENIZER_MODEL} \
    --rm-tokenizer-models ${RM_TOKENIZER_MODELS} \
    --dataloader-type external \
    --vocab-file none \
    --merge-file none \
    --num-workers 1 \
    --gdatasetv4-train-metadata-file ${data_path} \
    --px-shuffle-data \
    --px-shuffle-buffer-size ${SHUFFLE_BUFFER_SIZE} \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval -1 \
    --tensorboard-dir $TB_DIR \
    --tensorboard-log-interval 1 \
    --eval-interval 1 \
    --eval-iters $EVAL_ITERS \
"

if [ "$PPO_ROLE" = "actor" ]; then
    OUTPUT_ARGS="$OUTPUT_ARGS
        --wandb-project $WANDB_PROJECT \
        --wandb-exp-name $WANDB_EXP_NAME \
        --wandb-save-dir wandb \
    "
fi

GEN_ARGS="
    --ppo-sort-prompts-across-batches 8 \
    --ppo-rollout-max-prompt-len-diff 128 \
"


RL_ARGS="$GEN_ARGS
    --ppo-early-swap-model \
    --infer-engine-impl sglang \
    --ppo-auto-calc-args \
    --distributed-timeout-minutes $DIST_TIMEOUT_MIN \
    --ppo-display-rollout-generation \
    --ppo-disable-tqdm \
    --ppo-standalone-sampler \
    --use-gen-rm \
    --no-use-rm-and-critic \
    --hf-config-json-path $TOKENIZER_MODEL/config.json \
    --ppo-actor-node-ips $ACTOR_NODE_IPS \
    --ppo-actor-data-parallel-size $ACTOR_DP_SIZE \
    --ppo-actor-pipeline-model-parallel-size $ACTOR_PP_SIZE \
    --ppo-gen-rm-ips $GEN_RM_SVR_IPS \
    --ppo-gen-rm-ports $GEN_RM_SVR_PORTS \
    --ppo-gen-rm-pipeline-model-parallel-size $GEN_RM_PP_SIZE \
    --ppo-gen-rm-tensor-model-parallel-size $GEN_RM_TP_SIZE \
    --ppo-gen-rm-data-parallel-size $GEN_RM_DP_SIZE \
    --gen-rm-dist-init-addrs $GEN_RM_DIST_INIT_ADDRS \
    --ppo-sampler-ips $SAMPLER_SVR_IPS \
    --ppo-sampler-ports $SAMPLER_SVR_PORTS \
    --sampler-dist-init-addrs $SAMPLER_DIST_INIT_ADDRS \
    --ppo-sampler-tensor-model-parallel-size $SAMPLER_TP_SIZE \
    --ppo-sampler-pipeline-model-parallel-size $SAMPLER_PP_SIZE \
    --ppo-sampler-data-parallel-size $SAMPLER_DP_SIZE \
    --ppo-step-update-sampler-interval 1 \
    --ppo-max-epochs 2 \
    --ppo-max-epochs-2 2 \
    --ppo-step-save-interval 10 \
    --ppo-step-per-epoch -1 \
    --ppo-rollout-micro-batch-size $ROLLOUT_MICRO_BATCH_SIZE \
    --ppo-rollout-global-batch-size $ROLLOUT_GLOBAL_BATCH_SIZE \
    --ppo-resp-seq-len $RESP_SEQ_LENGTH \
    --ppo-rollout-pad-to-multiple-of 1024 \
    --ppo-logps-fwd-micro-batch-size $PPO_LOGPS_FWD_MICRO_BATCH_SIZE \
    --combine-rm-and-critic-server \
    --ppo-rollout-top-p 0.9 \
    --ppo-rollout-top-k 0 \
    --ppo-rollout-temperature 1.0 \
    --ppo-gen-rm-top-p 0.9 \
    --ppo-gen-rm-top-k 0 \
    --ppo-gen-rm-temperature 0.7 \
    --ppo-ratio-eps 0.2 \
    --ppo-rm-mask-prompt \
    --rm-output-scalar 1 \
    --rm-output-sequence 0 \
    --ppo-sampling-keeping-strategy all \
    --ppo-sampling-repeat 4 \
    --ppo-sampling-keep 4 \
    --ppo-use-absolute-kl \
    --use-grpo \
    --grpo-advantage-epsilon 1e-4 \
    --grpo-kl-loss-beta 1e-2 \
    --rm-head-arch multi_layers \
    --ppo-save-first-rollout-data \
    --ppo-grpo-reward-type rm_with_rule \
    --gen-term-at-nan \
    --ppo-rm-reward-alpha 0.5 \
    --ppo-rule-reward-beta 0.5 \
    --grpo-prefetch-samplings \
    --ppo-dynamic-sampling-max-replay $MAX_SAMPLING_RETRIES \
    --update-weight-max-size-mb 512 \
    --sampler-gpu-memory-utilization 0.7 \
    --gen-rm-gpu-memory-utilization 0.7 \
    --ppo-smart-pad-infer \
    --ppo-smart-pad-train \
    --ppo-grpo-reward-type rm_only \
    --ppo-gen-rm-repeat 8 \
    --ppo-gen-rm-resp-seq-len 512 \
    --ppo-num-rm 2 \
"

if [ "$PPO_ROLE" = "sampler" ] || [ "$PPO_ROLE" = "gen-rm" ]; then
  RL_ARGS="$RL_ARGS
      --no-fused-kernel \
  "
fi

if [ "$PPO_ROLE" = "gen-rm" ]; then
  RL_ARGS="$RL_ARGS
      --rsync-ckpts-to-shm \
  "
fi

if [ "$PPO_ROLE" = "actor" ]; then
  RL_ARGS="$RL_ARGS
      --ppo-actor-freeze-ppo-steps 0 \
  "
fi

# 热启动的时候去掉 FINETUNE_ARGS args
FINETUNE_ARGS="
    --finetune \
"

ACTOR_NODE_IPS=($ACTOR_NODE_IPS)
MONITOR_ARGS="
  --do-monitor \
  --monitor-server-ip ${ACTOR_NODE_IPS[0]} \
  --monitor-port 62000 \
  --auto-set-finetune-arg \
"
if [ "$PPO_ROLE" = "actor" ]; then
    RUN_PY='./tasks/math_rl_v3/train_ppo_actor.py'
elif [ "$PPO_ROLE" = "gen-rm" ]; then
    RUN_PY='./tasks/math_rl_v3/train_ppo_gen_rm.py'
elif [ "$PPO_ROLE" = "sampler" ]; then
    RUN_PY='./tasks/math_rl_v3/train_ppo_sampler.py'
else
    echo "no ${PPO_ROLE}"
    exit 0
fi

torchrun $DISTRIBUTED_ARGS $RUN_PY \
    $MP_ARGS \
    $TRAINING_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $RL_ARGS \
    $FINETUNE_ARGS \
    $MONITOR_ARGS \
    --cli-arg-yaml-cfgs $MODEL_YAML \
    --distributed-backend nccl \
    --save $SAVE_CHECKPOINT_DIR \
    --load $LOAD_CHECKPOINT_DIR \
    --load-ref $REF_LOAD_CHECKPOINT_DIR
