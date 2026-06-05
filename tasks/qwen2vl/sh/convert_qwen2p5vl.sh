# !/bin/bash
# $1: convert type: hf_to_mlm/mlm_to_hf
# $2: train type: sft/dpo/lora

readonly CONVERT_TYPE=$1
readonly TRAIN_TYPE=${2:-sft}

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=65535
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PX_INSPECET_MODEL=1

MYWD=$PWD
readonly HF_HUB_DIR="$MYWD/hf-hub/Qwen/Qwen2.5-VL-3B-Instruct"

if [ "$CONVERT_TYPE" == "hf_to_mlm" ]; then
    HF_INPUT_DIR=$HF_HUB_DIR
    MLM_OUTPUT_DIR="${PWD}/ckpt_qwen2p5vl_${TRAIN_TYPE}"

    ARGS="
        --model_arch qwen2vl \
        --convert_way hf_to_mlm \
        --megatron_load_dir xxx \
        --megatron_save_dir ${MLM_OUTPUT_DIR} \
        --hf_load_dir ${HF_INPUT_DIR} \
        --hf_save_dir xxx \
        --hf_py_source_file null \
        --tokenizer_type HuggingFaceTokenizer \
        --tokenizer_path ${HF_HUB_DIR} \
        --hf_config_json ${HF_HUB_DIR}/config.json \
        --bf16 \
        --dist_ckpt_format torch_dist \
    "
elif [ "$CONVERT_TYPE" == "mlm_to_hf" ]; then
    MLM_INPUT_DIR="${PWD}/ckpt_qwen2p5vl_${TRAIN_TYPE}/release"
    HF_OUTPUT_DIR="${PWD}/ckpt_qwen2p5vl_${TRAIN_TYPE}/hf_release"

    ARGS="
        --model_arch qwen2vl \
        --convert_way mlm_to_hf \
        --megatron_load_dir ${MLM_INPUT_DIR} \
        --megatron_save_dir xxx \
        --hf_load_dir xxx \
        --hf_save_dir ${HF_OUTPUT_DIR} \
        --hf_py_source_file null \
        --tokenizer_type HuggingFaceTokenizer \
        --tokenizer_path ${HF_HUB_DIR} \
        --hf_config_json ${HF_HUB_DIR}/config.json \
        --bf16 \
        --dist_ckpt_format torch_dist \
    "
else
    echo "not support this convert type:${CONVERT_TYPE}"
    exit 0
fi

if [ $TRAIN_TYPE == "sft" ]; then
    LORA_ARGS=""
    DPO_ARGS=""
elif [ $TRAIN_TYPE == "dpo" ]; then
    LORA_ARGS=""

    readonly REF_HF_ORIGIN_DIR=$HF_HUB_DIR
    DPO_ARGS="
        --qwen2vl_dpo \
        --qwen2vl_dpo_hf_ref_model ${REF_HF_ORIGIN_DIR} \
        --qwen2vl_dpo_choice_model policy \
    "
elif [ $TRAIN_TYPE == "lora" ]; then
    LORA_ARGS="
        --enable_lora \
        --lora_r 128 \
    "
    DPO_ARGS=""
else
    echo "not support this train type:${TRAIN_TYPE}"
    exit 0
fi


readonly MLM_PATH="../3rdparty/Megatron-LM:../Megatron-LM"
export PYTHONPATH="$PWD:$MLM_PATH:$PYTHONPATH"

python3 tools/px_ckpt_conv/convert_qwen2p5vl.py \
    $ARGS \
    $LORA_ARGS \
    $DPO_ARGS
