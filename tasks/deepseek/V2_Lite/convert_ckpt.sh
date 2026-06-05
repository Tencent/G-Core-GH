readonly MCORE_PATH='../../Megatron-LM'
export PYTHONPATH="$PWD:$MCORE_PATH:$PYTHONPATH"

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=65535
# export PX_INSPECET_MODEL=1

# readonly DFS_PATH="/mnt/ceph-sz2-csp/mm-base-plt2"
readonly DFS_PATH="/mnt/ceph-hz1-csp/mm-base-plt2"
readonly HF_HUB_DIR="$DFS_PATH/nrwu/hf-hub/deepseek-ai/DeepSeek-V2-Lite"

if [ $1 == "hf_to_mlm" ]; then
    HF_INPUT_DIR="$DFS_PATH/nrwu/hf-hub/deepseek-ai/DeepSeek-V2-Lite"
    MLM_OUTPUT_DIR="${PWD}/deepseek-v2-lite-mlm"

    ARGS="
        --model_arch deepseek-v2-lite \
        --convert_way hf_to_mlm \
        --megatron_load_dir xxx \
        --megatron_save_dir ${MLM_OUTPUT_DIR} \
        --hf_load_dir ${HF_INPUT_DIR} \
        --hf_save_dir xxx \
        --tokenizer_path ${HF_HUB_DIR} \
        --hf_py_source_file ${HF_HUB_DIR} \
        --hf_config_json ${HF_HUB_DIR}/config.json \
        --bf16 \
        --moe_grouped_gemm \
        --use_te_grouped_gemm \
        --dist_ckpt_format torch_dist \
        --mlm_model_provider_module_name pretrain_gpt \
    "
elif [ $1 == "mlm_to_hf" ]; then
    MLM_INPUT_DIR="${PWD}/deepseek-v2-lite-mlm/release"
    HF_OUTPUT_DIR="${PWD}/deepseek-v2-lite-hf_release"

    ARGS="
        --model_arch deepseek-v2-lite \
        --convert_way mlm_to_hf \
        --megatron_load_dir $MLM_INPUT_DIR \
        --megatron_save_dir xxx \
        --hf_load_dir xxx \
        --hf_save_dir $HF_OUTPUT_DIR \
        --hf_py_source_file ${HF_HUB_DIR} \
        --tokenizer_path ${HF_HUB_DIR} \
        --hf_config_json ${HF_HUB_DIR}/config.json \
        --bf16 \
        --moe_grouped_gemm \
        --use_te_grouped_gemm \
        --dist_ckpt_format torch_dist \
        --mlm_model_provider_module_name pretrain_gpt \
    "
fi

python3 tools/px_ckpt_conv/px_ckpt_conv.py $ARGS
