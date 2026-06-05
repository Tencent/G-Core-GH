export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=65535
export PX_INSPECET_MODEL=1

readonly HF_HUB_DIR="$PWD/hf-hub/Qwen/Qwen2.5-3B-Instruct"

if [ $1 == "hf_to_mlm" ]; then
    HF_INPUT_DIR="$PWD/hf-hub/Qwen/Qwen2.5-3B-Instruct"
    MLM_OUTPUT_DIR="${PWD}/qwen_2_5_3b_instruct"

    ARGS="
        --model_arch qwen2.5-3b-instruct \
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
    MLM_INPUT_DIR="${PWD}/qwen_2_5_3b_instruct_save_tool_multi_round/iter_0003712"
    HF_OUTPUT_DIR="${PWD}/qwen_2_5_3b_instruct_save_tool_multi_round/hf_3712"

    ARGS="
        --model_arch qwen2.5-3b-instruct \
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
export PYTHONPATH="$PWD:$MLM_PATH:$PYTHONPATH"
python tools/px_ckpt_conv/px_ckpt_conv.py $ARGS
