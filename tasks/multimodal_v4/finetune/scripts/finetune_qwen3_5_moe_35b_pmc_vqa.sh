MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MLM_BRIDGE_PATH="/root/Megatron-Bridge/src"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MLM_BRIDGE_PATH:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_DEVICE_MAX_CONNECTIONS=1

ray stop --force

cp /etc/mpi/hostfile /root/hostfile
sed -i 's/slots=8/slots=1/g' /root/hostfile
mpirun -v --allow-run-as-root \
  --bind-to none --map-by slot --hostfile /root/hostfile \
  --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
  hostname

mpirun -v --allow-run-as-root \
  --bind-to none --map-by slot --hostfile /root/hostfile \
  --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
  pkill -9 -f python

DATASET_NAME="filter_4k_qwen3_5"
# 如果要带着跑的话，仅需要在第一次运行的时候执行，要跑的数据集过滤完一次之后就可以复用了，避免重复执行
# bash tasks/multimodal_v4/finetune/scripts/preprocess_qwen3_5_data_v4.sh

source tasks/multimodal_v4/finetune/scripts/mpirun_run_once.sh \
    tasks/multimodal_v4/finetune/scripts/run_once.sh

source tasks/multimodal_v4/finetune/scripts/mpirun_run_once.sh \
    tasks/multimodal_v4/finetune/scripts/lmdb_run.sh $DATASET_NAME

python3 -u gpatch_v4/entry/train_vlm_finetune.py \
    --config-path="../../tasks/multimodal_v4/finetune/yaml" \
    --config-name="qwen3_5_moe_sft.yaml"
