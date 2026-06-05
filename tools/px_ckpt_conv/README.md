# PX CKPT CONV

`px_ckpt_conv` 工具提供了大量的模型转换能力，除了 HF 与 MCore 的相互转换（是的，不限于 MLM），也包括了
MLM 内 RL 各种 stage 之间的转换。

## HF to MLM

以 baichuan-7B 为例：

```bash
readonly WORK_DIR="${PWD}"
readonly RUN_PY="${WORK_DIR}/tools/px_ckpt_conv/px_ckpt_conv.py"

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=65535

readonly HF_CKPT="/home/nrwu/work/data2/hf-hub/baichuan-inc/Baichuan-7B/"

ARGS="
    --model_arch baichuan-7b \
    --convert_way hf_to_mlm \
    --megatron_load_dir $PWD/ckpt-mlm \
    --megatron_save_dir $PWD/ckpt-mlm \
    --hf_load_dir $HF_CKPT \
    --hf_save_dir $HF_CKPT \
    --hf_py_source_file $HF_CKPT \
    --tokenizer_type HfAutoTokenizer \
    --tokenizer_path $HF_CKPT \
    --hf_config_json $HF_CKPT/config.json \
    --bf16 \
    --dist_ckpt_format zarr \
"

PYTHONPATH="${WORK_DIR}:${PYTHONPATH}" python $RUN_PY $ARGS
```

目前支持 `--model_arch` 包括
- `--model_arch llama`
- `--model_arch bog`
- `--model_arch baichuan-7b`
- `--model_arch welm_19b`
- `--model_arch yi-9b`
- `--model_arch qwen2-72b`

## MLM to HF

同样以 baichuan 7B 为例：

```bash
readonly WORK_DIR="${PWD}"
readonly RUN_PY="${WORK_DIR}/tools/px_ckpt_conv/px_ckpt_conv.py"

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=65535

readonly HF_CKPT="/home/nrwu/work/data2/hf-hub/baichuan-inc/Baichuan-7B/"
readonly MLM_CKPT="$PWD/ckpt-mlm/release"
readonly OUT_HF_CKPT="out-ckpt-hf"

ARGS="
    --model_arch baichuan-7b \
    --convert_way mlm_to_hf \
    --megatron_load_dir $MLM_CKPT \
    --megatron_save_dir $MLM_CKPT \
    --hf_load_dir $OUT_HF_CKPT \
    --hf_save_dir $OUT_HF_CKPT \
    --hf_py_source_file $HF_CKPT \
    --tokenizer_type HfAutoTokenizer \
    --tokenizer_path $HF_CKPT \
    --hf_config_json $HF_CKPT/config.json \
    --bf16 \
    --dist_ckpt_format zarr \
"
PYTHONPATH="${WORK_DIR}:${PYTHONPATH}" python -u $RUN_PY $ARGS
```

支持 `--model_arch` 同上。

## PPO 互转

TODO：补文档...
