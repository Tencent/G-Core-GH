#!/bin/bash

readonly MPI_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly MPI_SIZE="${OMPI_COMM_WORLD_SIZE:-1}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME="bond1"
export GLOO_SOCKET_IFNAME="bond1"
export NCCL_DEBUG=WARN
export RAY_DEDUP_LOGS=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# 设置 max_split_size_mb 避免切分过大的显存 block 导致显存碎片
# 默认值是 -1，表示任意大小的显存 block 都可以被切分
# 如果显存碎片问题严重到导致 CUDA OOM，可尝试设为 256 (empirical)
# max_split_size_mb 越小，越能缓解显存碎片问题，但性能可能会下降，建议不要设置太小，并且观察吞吐表现
# see https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:-1
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

DEVICE_TYPE='cuda'

readonly NODE_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly NNODES="${OMPI_COMM_WORLD_SIZE:-1}"
readonly GPUS_PER_NODE=${GPUS_PER_NODE:-8}
readonly WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

readonly TOPO_CONFIG_FOLDER=$1
readonly PPO_ROLE=$2

MYWD=$PWD
readonly HF_HUB_DIR="$MYWD/hf-hub/Qwen/Qwen2.5-VL-3B-Instruct"
readonly MLM_PATH=../Megatron-LM
export PYTHONPATH="$MLM_PATH:$PYTHONPATH"

SAMPLER_NNODES=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn sampler-nnodes`
SAMPLER_MASTER_ADDR=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn sampler-master-addr`
SAMPLER_SVR_IPS=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn sampler-svr-ips`
SAMPLER_SVR_PORTS=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn sampler-svr-ports`
SAMPLER_DIST_INIT_ADDRS=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn sampler-dist-init-addrs`
SAMPLER_TP_SIZE=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn sampler-tp-size`
SAMPLER_PP_SIZE=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn sampler-pp-size`

CRITIC_NNODES=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn critic-nnodes`
CRITIC_MASTER_ADDR=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn critic-master-addr`
CRITIC_SVR_IPS=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn critic-svr-ips`
CRITIC_SVR_PORTS=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn critic-svr-ports`
CRITIC_TP_SIZE=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn critic-tp-size`
CRITIC_PP_SIZE=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn critic-pp-size`

ACTOR_NNODES=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn actor-nnodes`
ACTOR_MASTER_ADDR=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn actor-master-addr`
ACTOR_SVR_IPS=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn actor-svr-ips`
ACTOR_SVR_PORTS=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn actor-svr-ports`
ACTOR_NODE_IPS=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn actor-node-ips`
ACTOR_TP_SIZE=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn actor-tp-size`
ACTOR_PP_SIZE=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn actor-pp-size`
ACTOR_CP_SIZE=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn actor-cp-size`

if [ "$PPO_ROLE" = "actor" ]; then
  export MASTER_ADDR="${ACTOR_MASTER_ADDR:-localhost}"
  readonly MASTER_PORT=65201
elif [ "$PPO_ROLE" = "critic" ]; then
  export MASTER_ADDR="${CRITIC_MASTER_ADDR:-localhost}"
  readonly MASTER_PORT=65205
elif [ "$PPO_ROLE" = "sampler" ]; then
  export MASTER_ADDR="${SAMPLER_MASTER_ADDR:-localhost}"
  readonly MASTER_PORT=65204
else
  echo "$PPO_ROLE no support"
  exit 0
fi

if [ "$PPO_ROLE" = "actor" ]; then
  readonly REF_LOAD_CHECKPOINT_DIR="${PWD}/ckpt_qwen2p5vl_sft"
  readonly LOAD_CHECKPOINT_DIR="${PWD}/ckpt_qwen2p5vl_sft"
  readonly SAVE_CHECKPOINT_DIR="${PWD}/ckpt_qwen2p5vl_actor_save"
  readonly TB_DIR="tb/grpo-actor"
  readonly WANDB_DIR="wandb_local/grpo-actor"
  readonly LR=1e-6
elif [ "$PPO_ROLE" = "sampler" ]; then
  readonly LOAD_CHECKPOINT_DIR=$HF_HUB_DIR
  readonly SAVE_CHECKPOINT_DIR="none"
  readonly REF_LOAD_CHECKPOINT_DIR=$LOAD_CHECKPOINT_DIR
  readonly TB_DIR="tb/dqa-ppo-sampler"
  readonly WANDB_DIR="wandb_local/grpo-sampler"
  readonly LR=0
else
  readonly REF_LOAD_CHECKPOINT_DIR="none"
  readonly LOAD_CHECKPOINT_DIR="none"
  readonly SAVE_CHECKPOINT_DIR="none"
  readonly TB_DIR="tb/grpo-critic"
  readonly WANDB_DIR="wandb_local/grpo-critic"
  readonly LR=0
fi

readonly LMDB_PORT=8312
readonly DATASET_ROOT="${MYWD}/hf-hub/hiyouga/geometry3k/gcore-data"
readonly LMDB_PATH="${DATASET_ROOT}/img_file.lmdb"
readonly TRAIN_DATASET_META="/tmp/filter_4k_geo3k_train_gdataset_v4.json"
readonly EVAL_DATASET_META="/tmp/filter_4k_geo3k_eval_gdataset_v4.json"
if [ "$PPO_ROLE" = "actor" ]; then
    python tools/data_convert/build_dataset_v4_meta.py \
        --name "hiyouga/geometry3k" \
        --description "the dataset from hiyouga/geometry3k" \
        --lmdb_port $LMDB_PORT \
        --output_fullpath $TRAIN_DATASET_META \
        --json_inputs $DATASET_ROOT/filter_4k/train-00000-of-00001.jsonl \
        --rebuild

    python tools/data_convert/build_dataset_v4_meta.py \
        --name "hiyouga/geometry3k" \
        --description "the dataset from hiyouga/geometry3k" \
        --lmdb_port $LMDB_PORT \
        --output_fullpath $EVAL_DATASET_META \
        --json_inputs $DATASET_ROOT/filter_4k/validation-00000-of-00001.jsonl \
        --rebuild
fi
DATASET_ARGS="
    --gdatasetv4-train-metadata-file $TRAIN_DATASET_META \
    --gdatasetv4-eval-metadata-file $EVAL_DATASET_META \
    --px-inputs-pad-to-longest \
"

ACTOR_TOKENIZER_MODEL=$HF_HUB_DIR
RM_TOKENIZER_MODELS=$HF_HUB_DIR

EXTRA_ARGS=""

if [ "$PPO_ROLE" = "critic" ]; then
  readonly MODEL_YAML="gpatch/model_yamls/qwen2.5-vl-3b.yaml"
  readonly TOKENIZER_MODEL=$RM_TOKENIZER_MODELS
elif [ "$PPO_ROLE" = "sampler" ]; then
  readonly MODEL_YAML="gpatch/model_yamls/qwen2.5-vl-3b.yaml"
  readonly TOKENIZER_MODEL=$HF_HUB_DIR
elif [ "$PPO_ROLE" = "actor" ]; then
  readonly MODEL_YAML="gpatch/model_yamls/qwen2.5-vl-3b.yaml"
  readonly TOKENIZER_MODEL=$ACTOR_TOKENIZER_MODEL
fi

readonly CRITIC_DP_SIZE=$(($GPUS_PER_NODE*$CRITIC_NNODES/$CRITIC_TP_SIZE/$CRITIC_PP_SIZE))
readonly SAMPLER_DP_SIZE=$(($GPUS_PER_NODE*$SAMPLER_NNODES/$SAMPLER_TP_SIZE/$SAMPLER_PP_SIZE))
readonly ACTOR_DP_SIZE=$(($GPUS_PER_NODE*$ACTOR_NNODES/$ACTOR_TP_SIZE/$ACTOR_PP_SIZE/$ACTOR_CP_SIZE))

if [ "$PPO_ROLE" = "critic" ]; then
  readonly TP_SIZE=$CRITIC_TP_SIZE
  readonly PP_SIZE=$CRITIC_PP_SIZE
  readonly EP_SIZE=1
  readonly CP_SIZE=1
  readonly DP_SIZE=$CRITIC_DP_SIZE
elif [ "$PPO_ROLE" = "sampler" ]; then
  readonly TP_SIZE=$SAMPLER_TP_SIZE
  readonly PP_SIZE=$SAMPLER_PP_SIZE
  readonly EP_SIZE=1
  readonly CP_SIZE=1
  readonly DP_SIZE=$SAMPLER_DP_SIZE
elif [ "$PPO_ROLE" = "actor" ]; then
  readonly TP_SIZE=$ACTOR_TP_SIZE
  readonly PP_SIZE=$ACTOR_PP_SIZE
  readonly EP_SIZE=1
  readonly CP_SIZE=$ACTOR_CP_SIZE
  readonly DP_SIZE=$ACTOR_DP_SIZE
fi

readonly ROLLOUT_GLOBAL_BATCH_SIZE=512
readonly ROLLOUT_MICRO_BATCH_SIZE=1
readonly ROLLOUT_MAX_TOKENS_TO_OOM=$((32*16*1024))
if [ "$PPO_ROLE" = "actor" ]; then
  readonly GLOBAL_BATCH_SIZE=128
  readonly MICRO_BATCH_SIZE=1
else
  # sampler 和 rm 不需要训练，没有真正的 GBS，设置成 dp size，方便 scaling。
  readonly GLOBAL_BATCH_SIZE=$DP_SIZE
  readonly MICRO_BATCH_SIZE=1
fi

readonly PPO_LOGPS_FWD_MICRO_BATCH_SIZE=1
readonly TRAIN_ITERS=-1
readonly EVAL_ITERS=0
readonly SEQ_LENGTH=4096
readonly RESP_SEQ_LENGTH=$((2048))
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
    --optimizer adam \
    --lr $LR \
    --min-lr 0 \
    --lr-decay-style constant \
    --weight-decay 0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-08 \
    --clip-grad 1.0 \
    --lr-warmup-iters 0 \
    --train-iters ${TRAIN_ITERS} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --attention-softmax-in-fp32 \
    --transformer-impl transformer_engine \
    --use-rotary-position-embeddings \
    --rotary-percent 1.0 \
    --rotary-seq-len-interpolation-factor 1 \
    --no-rope-fusion \
    --seq-length $SEQ_LENGTH \
    --decoder-seq-length $SEQ_LENGTH \
    --seed 42 \
    --ckpt-format torch_dist \
    --processor-path ${TOKENIZER_MODEL} \
    --mask-history \
    --image-token-id 151655 \
"

DATA_ARGS="
    $DATASET_ARGS \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --dataloader-type external \
    --num-workers 8 \
    --timing-log-level 1 \
    --px-dataloader-prefetch-factor 32 \
    --px-shuffle-buffer-size 102400 \
    --px-smart-padding-buffer-size 512 \
    --ppo-smart-pad-infer \
    --ppo-smart-pad-train \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval -1 \
    --tensorboard-dir $TB_DIR \
    --tensorboard-log-interval 1 \
    --eval-interval 1 \
    --eval-iters $EVAL_ITERS \
"

GEN_ARGS="
    --ppo-sort-prompts-across-batches 8 \
    --ppo-rollout-max-prompt-len-diff 128 \
    --max-tokens-to-oom $ROLLOUT_MAX_TOKENS_TO_OOM \
"

readonly DIST_TIMEOUT_MIN=300


RL_ARGS="$GEN_ARGS
    --ppo-auto-calc-args \
    --distributed-timeout-minutes $DIST_TIMEOUT_MIN \
    --ppo-display-rollout-generation \
    --ppo-disable-tqdm \
    --ppo-standalone-sampler \
    --infer-engine-impl sglang \
    --hf-config-json-path $TOKENIZER_MODEL/config.json \
    --ppo-actor-node-ips $ACTOR_NODE_IPS \
    --ppo-actor-data-parallel-size $ACTOR_DP_SIZE \
    --ppo-actor-pipeline-model-parallel-size $ACTOR_PP_SIZE \
    --ppo-critic-ips $CRITIC_SVR_IPS \
    --ppo-critic-ports $CRITIC_SVR_PORTS \
    --ppo-critic-pipeline-model-parallel-size $CRITIC_PP_SIZE \
    --ppo-critic-tensor-model-parallel-size $CRITIC_TP_SIZE \
    --ppo-critic-data-parallel-size $CRITIC_DP_SIZE \
    --ppo-sampler-ips $SAMPLER_SVR_IPS \
    --ppo-sampler-ports $SAMPLER_SVR_PORTS \
    --sampler-dist-init-addrs $SAMPLER_DIST_INIT_ADDRS \
    --ppo-sampler-tensor-model-parallel-size $SAMPLER_TP_SIZE \
    --ppo-sampler-pipeline-model-parallel-size $SAMPLER_PP_SIZE \
    --ppo-sampler-data-parallel-size $SAMPLER_DP_SIZE \
    --ppo-step-update-sampler-interval 1 \
    --ppo-max-epochs 15 \
    --ppo-max-epochs-2 1 \
    --ppo-step-save-interval 10 \
    --ppo-step-per-epoch -1 \
    --ppo-rollout-micro-batch-size $ROLLOUT_MICRO_BATCH_SIZE \
    --ppo-rollout-global-batch-size $ROLLOUT_GLOBAL_BATCH_SIZE \
    --ppo-resp-seq-len $RESP_SEQ_LENGTH \
    --ppo-rollout-pad-to-multiple-of 128 \
    --ppo-logps-fwd-micro-batch-size $PPO_LOGPS_FWD_MICRO_BATCH_SIZE \
    --combine-rm-and-critic-server \
    --ppo-rollout-top-p 1 \
    --ppo-rollout-top-k 0 \
    --ppo-rollout-temperature 1.0 \
    --ppo-ratio-eps 0.2 \
    --ppo-rm-mask-prompt \
    --rm-output-scalar 1 \
    --rm-output-sequence 0 \
    --ppo-sampling-repeat 5 \
    --ppo-sampling-keep 5 \
    --ppo-use-absolute-kl \
    --use-grpo \
    --grpo-advantage-epsilon 1e-4 \
    --grpo-kl-loss-beta 1e-2 \
    --rm-head-arch multi_layers \
    --ppo-save-first-rollout-data \
    --ppo-grpo-reward-type rule_only \
    --gen-term-at-nan \
    --ppo-rm-reward-alpha 0.5 \
    --ppo-rule-reward-beta 0.5 \
    --grpo-prefetch-samplings \
    --ppo-dynamic-sampling-max-replay $MAX_SAMPLING_RETRIES \
    --ppo-dual-clip-ratio-c 1.2 \
    --px-use-fast-tokenizer \
    --ppo-early-swap-model \
    --ppo-clamp-kl-val 20 \
"

if [ "$PPO_ROLE" = "sampler" ]; then
  RL_ARGS="$RL_ARGS
      --no-fused-kernel \
  "
fi

if [ "$PPO_ROLE" = "critic" ]; then
  RL_ARGS="$RL_ARGS
      --no-fused-kernel \
  "
fi

# 热启动的时候去掉 FINETUNE_ARGS args
FINETUNE_ARGS="
    --finetune \
"

if [ "$PPO_ROLE" = "actor" ]; then
  RUN_PY='./tasks/qwen2vl/grpo/train_qwen2vl_ppo_actor.py'
  # 如果 actor 与 sampler 分离的话，sampler也得启动这个
  nohup python megatron_datasets/tools/lmdb_read_svr.py \
    --lmdb-path $LMDB_PATH \
    --lmdb-map-size 500 \
    --lmdb-port $LMDB_PORT > svr.log 2>&1 &
elif [ "$PPO_ROLE" = "critic" ]; then
  RUN_PY='./tasks/qwen2vl/grpo/train_qwen2vl_ppo_critic.py'
elif [ "$PPO_ROLE" = "sampler" ]; then
  # sampler的环境变量应该尽可能在ray start前设置
  # export VLLM_HOST_IP=`ifconfig bond1 | grep inet | grep broadcast | awk -F' ' '{print $2}'`
  # echo "VLLM_HOST_IP: $VLLM_HOST_IP"
  # python tools/auto_place.py --fn init_ray --config-folder $TOPO_CONFIG_FOLDER

  RUN_PY='./tasks/qwen2vl/grpo/train_qwen2vl_ppo_sampler.py'
else
  echo "no ${PPO_ROLE}"
  exit 0
fi

wandb offline
if [ "$PPO_ROLE" = "actor" ]; then
  EXTRA_ARGS+=""
  EXTRA_ARGS+=" --wandb-project qwen2vl-grpo "
  EXTRA_ARGS+=" --wandb-exp-name qwen2p5/3b/geometry3k/baseline "
  EXTRA_ARGS+=" --wandb-save-dir $WANDB_DIR "
elif [ "$PPO_ROLE" = "critic" ]; then
  EXTRA_ARGS+=" --ppo-mm-rule-type geometry3k "
fi

echo "extra args: $EXTRA_ARGS"

if [ "$PPO_ROLE" = "sampler" ] || [ "$PPO_ROLE" = "gen-rm" ]; then
    MP_ARGS="$MP_ARGS
        --use-tp-pp-dp-mapping \
    "
fi

echo "extra args: $EXTRA_ARGS"

EVAL_ARGS="
    --ppo-step-eval-interval 2 \
    --ppo-eval-steps 1 \
    --ppo-eval-rollout-global-batch-size 256 \
    --ppo-eval-rollout-micro-batch-size 1 \
"

torchrun $DISTRIBUTED_ARGS $RUN_PY \
    $TRAINER_ARGS \
    $MP_ARGS \
    $DATA_ARGS \
    $EVAL_ARGS \
    $OUTPUT_ARGS \
    $RL_ARGS \
    $FINETUNE_ARGS \
    $EXTRA_ARGS \
    --cli-arg-yaml-cfgs $MODEL_YAML \
    --distributed-backend nccl \
    --save $SAVE_CHECKPOINT_DIR \
    --load $LOAD_CHECKPOINT_DIR \
    --load-ref $REF_LOAD_CHECKPOINT_DIR

pkill -f -9 lmdb_read_svr.py
