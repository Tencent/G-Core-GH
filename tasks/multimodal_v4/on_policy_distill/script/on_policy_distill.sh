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
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
  hostname

mpirun -v --allow-run-as-root \
  --bind-to none --map-by slot --hostfile /root/hostfile \
  --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
  pkill -9 -f python

source tasks/multimodal_v4/on_policy_distill/script/mpirun_run_once.sh \
    tasks/multimodal_v4/on_policy_distill/script/run_once.sh

python3 -u gpatch_v4/entry/train_on_policy_distill.py \
    --config-path="../../tasks/multimodal_v4/on_policy_distill/yaml" \
    --config-name="qwen3vl_distill"
