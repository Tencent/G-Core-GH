set -ex

HOSTFILE="/root/hostfile"
cp /etc/mpi/hostfile $HOSTFILE
sed -i 's/slots=1/slots=8/g' $HOSTFILE # 每机器 8 进程，一次性测试脚本随意些

export _MASTER_ADDR=$__POD_IP__
export PYTHONPATH="$PWD:$PYTHONPATH"

readonly OUTPUT_DIR="eval_answer"
mkdir -p $OUTPUT_DIR

readonly MODEL_PATH="$PWD/qwen_2_5_1_5b_save/hf_3712/"
readonly OUTPUT_PATH="$OUTPUT_DIR/grpo_math_3712_eval"

DATA_PATH="$PWD/hf-hub/openai/gsm8k-jsonl/eval/test-00000-of-00001.jsonl"

mpirun -v --allow-run-as-root \
  --bind-to none --map-by slot --hostfile $HOSTFILE \
  --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x _MASTER_ADDR \
  python3 tasks/math_rl_v3/eval_model.py \
  --model_path $MODEL_PATH \
  --data_path $DATA_PATH \
  --output_path $OUTPUT_PATH
