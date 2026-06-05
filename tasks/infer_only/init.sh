MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$PYTHONPATH"

source tasks/infer_only/mpirun-init-ray.sh