# Tested on Megatron-Bridge commit 809201ff1451d36e8212948bb371fca2b07b3d57
PATH_TO_MEGATRON_BRIDGE="../3rdparty/Megatron-Bridge/"
PATH_TO_MEGATRON_DEV="../3rdparty/Megatron-LM/"
PATH_TO_GCORE_DEV="$PWD"
export PYTHONPATH="$PATH_TO_MEGATRON_BRIDGE/src:$PATH_TO_MEGATRON_BRIDGE:$PATH_TO_MEGATRON_DEV:$PATH_TO_GCORE_DEV:$PYTHONPATH"

pip install omegaconf
# pip install nvidia-modelopt[torch] # 这东西不能用，会导致镜像挂掉

if [ $1 == "hf_to_mlm" ]; then
    HF_INPUT_PATH=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/lmsys/gpt-oss-20b-bf16
    MLM_SAVE_PATH=./gpt-oss-20b-bf16-mlm
    python $PATH_TO_MEGATRON_BRIDGE/examples/conversion/convert_checkpoints.py import \
        --hf-model $HF_INPUT_PATH \
        --megatron-path $MLM_SAVE_PATH \
        --torch-dtype bfloat16 \
        --device-map auto \
        --trust-remote-code \

    # 修改存储的一些格式
    mv $MLM_SAVE_PATH/iter_0000000 $MLM_SAVE_PATH/iter_0000001
    echo 1 > $MLM_SAVE_PATH/latest_checkpointed_iteration.txt

    python tasks/gpt_oss/fake_common_pt.py --mlm-path $MLM_SAVE_PATH/iter_0000001
elif [ $1 == "mlm_to_hf" ]; then
    MLM_INPUT_PATH=/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_7/user_oriontian/llm/gpt_oss_sft_baseline_mlm_actor_sft_v2_v6_mix_d3s_cover_strategy_online_data_gpt_oss/iter_0001920
    HF_SAVE_PATH='./gpt-oss-120b-hf-iter1920'
    cp examples/astrachang/gpt-oss/ugly_bridge_yaml.yaml $MLM_INPUT_PATH/run_config.yaml
    python tasks/gpt_oss/rm_training_progress_common.py --mlm-path $MLM_INPUT_PATH
    python $PATH_TO_MEGATRON_BRIDGE/examples/conversion/convert_checkpoints.py export \
        --hf-model ../../../nrwu/hf-hub/lmsys/gpt-oss-120b-bf16/ \
        --hf-path $HF_SAVE_PATH \
        --megatron-path $MLM_INPUT_PATH
fi
