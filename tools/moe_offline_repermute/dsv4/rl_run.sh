MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"

export PYTHONPATH="$PWD:$PWD/tests:$PWD/tests/test_gpatch_v4:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# Change the config path and name to your actual task.
CONFIG_PATH="$PWD/tasks/math_dsv4/yaml"
CONFIG_NAME="math_dsv4_grpo_sgl_colocated.yaml"
COUNTS="debug-tmp/debug_dsv4_router/counts_rl.pt"
DST_CKPT="moe_repermute_rl/DeepSeek-V4-Flash-repermuted"
WORK_DIR="moe_repermute_rl"
EP_SIZES="[4,8,16]"
VERIFY=True
VERIFY_FULL_FORWARD=True
FORCE_REFRESH_AUX=True

# fails loudly if stage 1 failed to dump counts.pt
rm $COUNTS

# Stage 1 of the offline expert re-permutation pipeline:
# run one step, dump expert token counts, then sys.exit(0).
export GCORE_NNODES=16
source tasks/math_dsv4/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/train_lm_grpo.py \
    --config-path=$CONFIG_PATH --config-name=$CONFIG_NAME \
    training.auto_load_from_save_ckpt=False \
    +debug.debug_dump_expert_token_counts=True \
    +debug.debug_dump_expert_token_counts_path=$COUNTS \

sleep 10

# run the offline expert re-permutation pipeline
python3 -u -m tools.moe_offline_repermute.dsv4.repermute_pipeline_dsv4 \
    --config-path=$CONFIG_PATH --config-name=$CONFIG_NAME \
    +counts=$COUNTS \
    +dst_ckpt=$DST_CKPT \
    +work_dir=$WORK_DIR \
    +ep_sizes=$EP_SIZES \
    +verify=$VERIFY \
    +verify_full_forward=$VERIFY_FULL_FORWARD \
    +force_refresh_aux=$FORCE_REFRESH_AUX \
