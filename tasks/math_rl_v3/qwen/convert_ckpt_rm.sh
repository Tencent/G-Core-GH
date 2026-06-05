export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=65535
export PX_INSPECET_MODEL=1

readonly HF_HUB_DIR="$PWD/hf-hub/Qwen/Qwen2.5-Math-RM-72B"
readonly HF_PY_SRC="$HF_HUB_DIR"
readonly HF_DIR="$PWD/hf-hub/Qwen/Qwen2.5-Math-RM-72B"
readonly MLM_DIR="${PWD}/qwen_2_5_rm_72b"

ARGS="
    --model_arch qwen2.5-math-rm-72b \
    --convert_way hf_to_mlm \
    --megatron_load_dir xxx \
    --megatron_save_dir ${MLM_DIR} \
    --hf_load_dir ${HF_DIR} \
    --hf_save_dir xxx \
    --hf_py_source_file ${HF_PY_SRC} \
    --tokenizer_path ${HF_HUB_DIR} \
    --hf_config_json ${HF_PY_SRC}/config.json \
    --bf16 \
    --dist_ckpt_format torch_dist \
    --rm_multi_layers \
    --hf_auto_model_class_name AutoModel \
    --mlm_model_provider_module_name tasks.math_rl_v3.train_ppo_critic \
"

readonly MLM_PATH="../3rdparty/Megatron-LM:../Megatron-LM"
export PYTHONPATH="$PWD:$MLM_PATH:$PYTHONPATH"
python3 tools/px_ckpt_conv/convert_rm.py $ARGS
