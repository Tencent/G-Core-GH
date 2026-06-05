WORK_DIR="${PWD}" && readonly WORK_DIR
PROJECT_DIR="${WORK_DIR}" && readonly PROJECT_DIR
echo $PROJECT_DIR
readonly RUN_PY="${PROJECT_DIR}/tools/checkpoint/ckpt_conversion/ckpt_conversion.py"

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=65534

ckpt=("iter_0450000" "iter_0500000" "iter_0550000" "iter_0600000" "iter_0650000")
# ckpt=("iter_0400000")
for value in ${ckpt[@]}
do
ARGS="
    --model_arch llama \
    --convert_way mlm_to_hf \
    --megatron_load_dir /mnt/gemininjceph/geminicephfs/mm-base-plt2/user_xiaotaoliu/tasks/pretrain_bog050/bog050_1B_4k_ep2_0515/$value \
    --megatron_save_dir /data/xiaotaoliu/test_dist_ckpt/ \
    --hf_load_dir /mnt/gemininjceph/geminicephfs/mm-base-plt2/user_xiaotaoliu/tasks/pretrain_bog050/bog050_1B_4k_ep2_0515_hf/ \
    --hf_save_dir /mnt/gemininjceph/geminicephfs/mm-base-plt2/user_xiaotaoliu/tasks/pretrain_bog050/bog050_1B_4k_ep2_0515_hf/$value \
    --hf_py_source_file /mnt/gemininjceph/geminicephfs/mm-base-plt2/user_xiaotaoliu/continual_train/code/Megatron-LM5/tools/checkpoint/ckpt_conversion/bog_source_file/bog \
    --tokenizer_path /mnt/gemininjceph/geminicephfs/mm-base-plt2/user_xiaotaoliu/continual_train/code/Megatron-LM5/tools/checkpoint/ckpt_conversion/bog_source_file/pretrainx_tokenizer_v2_unigram_200k \
    --tokenizer_type PretrainxTokenizer-0.3-ulm \
    --hf_config_json /mnt/gemininjceph/geminicephfs/mm-base-plt2/user_xiaotaoliu/tasks/pretrain_bog050/bog050_1B_4k_ep2_0515_hf/config.json \
    --bf16 \
    --dist_ckpt_format zarr \
"
PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}" python $RUN_PY $ARGS
done

# ARGS="
#     --model_arch baichuan-7b \
#     --convert_way hf_to_mlm \
#     --megatron_load_dir /data/xiaotaoliu/test_dist_ckpt/ \
#     --megatron_save_dir /data/xiaotaoliu/test_dist_ckpt/ \
#     --hf_load_dir /data/xiaotaoliu/Baichuan-7B-w-pad  \
#     --hf_save_dir /data/xiaotaoliu/Baichuan-7B-w-pad  \
#     --hf_py_source_file /data/xiaotaoliu/Baichuan-7B-w-pad \
#     --tokenizer_type HfAutoTokenizer \
#     --tokenizer_path /data/xiaotaoliu/Baichuan-7B-w-pad \
#     --hf_config_json /data/xiaotaoliu/Baichuan-7B-w-pad/config.json \
#     --bf16 \
#     --dist_ckpt_format zarr \
#     --kv_channels 128 \
# "
# PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}" python $RUN_PY $ARGS
