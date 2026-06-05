MYWD=$PWD
export HF_HUB_DIR="$PWD/hf-hub/Qwen/Qwen3-VL-30B-A3B-Instruct"
export HF_HUB_DIR=/data/Qwen3-VL-30B-A3B-Instruct
export LOAD_CHECKPOINT_DIR=$HF_HUB_DIR
export SAVE_CHECKPOINT_DIR="$PWD/ckpt_qwen3vl_router_replay_save"
export TOKENIZER_MODEL=${HF_HUB_DIR}
export MODEL_YAML="$PWD/gpatch/model_yamls/qwen3vl-30b-a3b-moe.yaml"
export DATALOADER_ARGS="
--use-new-dataloader \
--template qwen3_vl \
--dataset-impl energon \
--dataset-dir demo_data \
--media-dir demo_data/images \
--dataset demo \
--max-samples 8192 \
"
export DISABLE_VERSION_CHECK=1
#export TOKENIZERS_PARALLELISM=1
readonly MLM_PATH=$MYWD/../Megatron-LM
readonly MBRIDGE_PATH=$MYWD/../mbridge
#readonly LLaMAFactory_Path=$MYWD/../LLaMA-Factory/src
#readonly EnergonPath=$MYWD/../Megatron-Energon/src
export PYTHONPATH="$MLM_PATH:$PYTHONPATH:${MBRIDGE_PATH}" #:${LLaMAFactory_Path}:${EnergonPath}"
#export CUDA_LAUNCH_BLOCKING=1
#export TORCH_USE_CUDA_DSA=1
bash tasks/router_replay/grpo/qwen3vl_grpo.sh $1 $2