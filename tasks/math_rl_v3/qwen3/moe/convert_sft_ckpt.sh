MCORE_PATH="/root/Megatron-LM"
export PYTHONPATH="$PWD:$MCORE_PATH:$PYTHONPATH"

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=65535
export PX_INSPECET_MODEL=1

DFS_PATH=$PWD
# DFS_PATH="/mnt/gemininjceph2/geminicephfs/mm-base-plt2"

# model: qwen3 moe 30B-A3B
readonly HF_HUB_DIR="$DFS_PATH/hf-hub/Qwen/Qwen3-30B-A3B/"

if [ $1 == "hf_to_mlm" ]; then
    # # model: qwen3 30B-A3B
    HF_INPUT_DIR="$DFS_PATH/hf-hub/Qwen/Qwen3-30B-A3B/"
    MLM_OUTPUT_DIR="${PWD}/qwen3_30b_a3b_moe_mlm"

    ARGS="
        --model_arch qwen3-moe \
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
        --moe_grouped_gemm \
        --use_te_grouped_gemm \
    "
elif [ $1 == "mlm_to_hf" ]; then
    # # model: qwen3 30B-A3B
    MLM_INPUT_DIR="${PWD}/qwen3_30b_a3b_moe_mlm/release"
    HF_OUTPUT_DIR="${PWD}/qwen3_30b_a3b_moe_mlm/hf_converted/"

    ARGS="
        --model_arch qwen3-moe \
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
        --moe_grouped_gemm \
        --use_te_grouped_gemm \
    "
fi


python3 tools/px_ckpt_conv/px_ckpt_conv.py $ARGS
