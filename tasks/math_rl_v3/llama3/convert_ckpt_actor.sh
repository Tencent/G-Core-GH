readonly MCORE_PATH='../3rdparty/Megatron-LM'
export PYTHONPATH="$PWD:$MCORE_PATH:$PYTHONPATH"

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=65535
export PX_INSPECET_MODEL=1

readonly DFS_PATH="/mnt/ceph-hz1-csp/mm-base-plt2"

readonly HF_HUB_DIR="$DFS_PATH/nrwu/hf-hub/unsloth/Llama-3.2-1B"
# readonly HF_HUB_DIR="$DFS_PATH/nrwu/hf-hub/unsloth/Llama-3.1-8B"

LOG_DIR="$PWD/log/math_rl_llama3"
mkdir -p $LOG_DIR

if [ $1 == "hf_to_mlm" ]; then
    HF_INPUT_DIR="$DFS_PATH/nrwu/hf-hub/unsloth/Llama-3.2-1B"
    MLM_OUTPUT_DIR="${PWD}/llama_3_2_1b"
    # HF_INPUT_DIR="$DFS_PATH/nrwu/hf-hub/unsloth/Llama-3.1-8B"
    # MLM_OUTPUT_DIR="${PWD}/llama_3_1_8b"

    ARGS="
        --model_arch llama \
        --convert_way hf_to_mlm \
        --megatron_load_dir xxx \
        --megatron_save_dir ${MLM_OUTPUT_DIR} \
        --hf_load_dir ${HF_INPUT_DIR} \
        --hf_save_dir xxx \
        --hf_py_source_file ${HF_HUB_DIR} \
        --tokenizer_path ${HF_HUB_DIR} \
        --hf_config_json ${HF_HUB_DIR}/config.json \
        --bf16 \
        --dist_ckpt_format torch_dist \
    "
elif [ $1 == "mlm_to_hf" ]; then
    MLM_INPUT_DIR="${PWD}/llama_3_2_1b_save/iter_0003712"
    HF_OUTPUT_DIR="${PWD}/llama_3_2_1b_save/hf_release"

    # rm with rule save ckpt
    # MLM_INPUT_DIR="${PWD}/llama_3_2_1b_save/iter_0003712"
    # HF_OUTPUT_DIR="${PWD}/llama_3_2_1b_save/hf_iter_0003712"

    ARGS="
        --model_arch llama \
        --convert_way mlm_to_hf \
        --megatron_load_dir $MLM_INPUT_DIR \
        --megatron_save_dir xxx \
        --hf_load_dir xxx \
        --hf_save_dir $HF_OUTPUT_DIR \
        --hf_py_source_file ${HF_HUB_DIR} \
        --tokenizer_path ${HF_HUB_DIR} \
        --hf_config_json ${HF_HUB_DIR}/config.json \
        --bf16 \
        --dist_ckpt_format torch_dist \
    "
fi

readonly MLM_PATH="../3rdparty/Megatron-LM:../Megatron-LM"
export PYTHONPATH="$MLM_PATH:$PYTHONPATH"
python3 tools/px_ckpt_conv/px_ckpt_conv.py $ARGS > $LOG_DIR/convert_ckpt.log 2>&1
