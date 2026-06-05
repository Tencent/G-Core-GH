# !/bin/bash
# $1: convert type: hf_to_mlm/mlm_to_hf

readonly CONVERT_TYPE=$1

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=65535
export CUDA_DEVICE_MAX_CONNECTIONS=1


MYWD=$PWD
readonly HF_HUB_DIR="$MYWD/hf-hub/OpenGVLab/InternVL3-2B"

# Deprecated
# only support 2B
if [ "$CONVERT_TYPE" == "hf_to_mlm" ]; then
    HF_INPUT_DIR=$HF_HUB_DIR
    MLM_OUTPUT_DIR="${PWD}/ckpt_internvl3_2B"

    ARGS="
        --model_arch internvl \
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
    MLM_INPUT_DIR="${PWD}/ckpt_internvl3_2B/release"
    HF_OUTPUT_DIR="${PWD}/ckpt_internvl3_2B/hf_release"

    ARGS="
        --model_arch internvl \
        --convert_way mlm_to_hf \
        --megatron_load_dir ${MLM_INPUT_DIR} \
        --megatron_save_dir xxx \
        --hf_load_dir xxx \
        --hf_save_dir ${HF_OUTPUT_DIR} \
        --hf_py_source_file ${HF_HUB_DIR} \
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

export PYTHONPATH=../Megatron-LM:../mbridge:$PYTHONPATH

readonly WORK_DIR="${PWD}"
readonly RUN_PY="${WORK_DIR}/tools/px_ckpt_conv/convert_internvl.py"

PYTHONPATH="${WORK_DIR}:${PYTHONPATH}" python $RUN_PY $ARGS
