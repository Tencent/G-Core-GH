# !/bin/bash
# $1: convert type: hf_to_mlm/mlm_to_hf
# $2: train type: sft/dpo

readonly CONVERT_TYPE=$1
readonly TRAIN_TYPE=${2:-sft}

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=65535
export CUDA_DEVICE_MAX_CONNECTIONS=1

MYWD=$PWD
readonly HF_HUB_DIR="$MYWD/hf-hub/google/gemma-3-4b-it"

if [ "$CONVERT_TYPE" == "hf_to_mlm" ]; then
    HF_INPUT_DIR=$HF_HUB_DIR
    MLM_OUTPUT_DIR="${PWD}/ckpt_gemma3_${TRAIN_TYPE}"

    ARGS="
        --model_arch gemma3 \
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
    MLM_INPUT_DIR="${PWD}/ckpt_gemma3_${TRAIN_TYPE}/release"
    HF_OUTPUT_DIR="${PWD}/ckpt_gemma3_${TRAIN_TYPE}/hf_release"

    ARGS="
        --model_arch gemma3 \
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
    DPO_ARGS=""
elif [ $TRAIN_TYPE == "dpo" ]; then
    readonly REF_HF_ORIGIN_DIR=$HF_HUB_DIR
    DPO_ARGS="
        --gemma3_dpo \
        --gemma3_dpo_hf_ref_model ${REF_HF_ORIGIN_DIR} \
        --gemma3_dpo_choice_model policy \
    "
else
    echo "not support this train type:${TRAIN_TYPE}"
    exit 0
fi

readonly MLM_PATH='../Megatron-LM'
export PYTHONPATH="$PWD:$MLM_PATH:$PYTHONPATH"

python tools/px_ckpt_conv/convert_gemma3.py \
    $ARGS \
    $DPO_ARGS
