export PYTHONPATH="$PWD:$PYTHONPATH"

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=65535
export PX_INSPECET_MODEL=1

DFS_PATH="/mnt/ceph-sz2-csp/mm-base-plt2"
# DFS_PATH="/mnt/gemininjceph2/geminicephfs/mm-base-plt2"

# model: qwen3 1.7B
readonly HF_HUB_DIR="$DFS_PATH/nrwu/hf-hub/Qwen/Qwen3-1.7B/"

# # model: qwen3 32B
# readonly HF_HUB_DIR="$DFS_PATH/nrwu/hf-hub/Qwen/Qwen3-32B/"

if [ $1 == "hf_to_mlm" ]; then
    # model: qwen3 1.7B
    HF_INPUT_DIR="$DFS_PATH/nrwu/hf-hub/Qwen/Qwen3-1.7B/"
    MLM_OUTPUT_DIR="${PWD}/qwen3_1-7b_mlm"

    # # model: qwen3 32B
    # HF_INPUT_DIR="$DFS_PATH/nrwu/hf-hub/Qwen/Qwen3-32B/"
    # MLM_OUTPUT_DIR="${PWD}/qwen3_32b_mlm"

    ARGS="
        --model_arch qwen3 \
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
        --kv_channels 128 \
    "
elif [ $1 == "mlm_to_hf" ]; then
    # # model: qwen3 1.7B
    MLM_INPUT_DIR="${PWD}/qwen3_1-7b_mlm/release"
    HF_OUTPUT_DIR="${PWD}/qwen3_1-7b_mlm/hf_converted/"

    # # model: qwen3 32B
    # MLM_INPUT_DIR="${PWD}/qwen3_32b_mlm/release"
    # HF_OUTPUT_DIR="${PWD}/qwen3_32b_mlm/hf_converted/"

    ARGS="
        --model_arch qwen3 \
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
        --kv_channels 128 \
    "
fi


python3 tools/px_ckpt_conv/px_ckpt_conv.py $ARGS
