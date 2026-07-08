# SRC_DIR="/mnt/ceph-hz1-csp/mm-base-plt2/user_xiaotaoliu/project/workspace2/vllm"
SRC_DIR="/mnt/geminihzceph/user_xiaotaoliu/project/workspace2/vllm"
DST_DIR="/root/vllm"

cp $SRC_DIR/vllm/model_executor/models/deepseek_v4.py $DST_DIR/vllm/model_executor/models/deepseek_v4.py
cp $SRC_DIR/vllm/model_executor/models/deepseek_v4_mtp.py $DST_DIR/vllm/model_executor/models/deepseek_v4_mtp.py
# cp $SRC_DIR/vllm/model_executor/layers/linear.py $DST_DIR/vllm/model_executor/layers/linear.py
# cp $SRC_DIR/vllm/model_executor/layers/quantization/fp8.py $DST_DIR/vllm/model_executor/layers/quantization/fp8.py
# cp $SRC_DIR/vllm/model_executor/layers/quantization/mxfp4.py $DST_DIR/vllm/model_executor/layers/quantization/mxfp4.py
# cp $SRC_DIR/vllm/model_executor/utils.py $DST_DIR/vllm/model_executor/utils.py
# cp $SRC_DIR/vllm/model_executor/layers/fused_moe/layer.py $DST_DIR/vllm/model_executor/layers/fused_moe/layer.py
cp $SRC_DIR/vllm/v1/worker/gpu_worker.py $DST_DIR/vllm/v1/worker/gpu_worker.py
cp $SRC_DIR/vllm/model_executor/parameter.py $DST_DIR/vllm/model_executor/parameter.py
cp $SRC_DIR/vllm/model_executor/layers/deepseek_v4_attention.py $DST_DIR/vllm/model_executor/layers/deepseek_v4_attention.py

# mpirun -v --allow-run-as-root   --bind-to none --map-by slot --hostfile /root/hostfile   --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct   -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x RAY_DEDUP_LOGS bash tasks/math_dsv4/scripts/temp_replace.sh

