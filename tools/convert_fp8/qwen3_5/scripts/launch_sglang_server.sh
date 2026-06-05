#!/bin/bash
set -euo pipefail

cd /work/wepsdl/gcore-dev

MODEL=/work/wepsdl/gcore-dev/Qwen3.6-35B-A3B-FP8-official-skip
PORT=30000
# Qwen3.6 MoE expert hidden size is 512; SGLang splits gate/up by TP before
# FP8 block setup, so TP=8 gives 64 which is not 128-aligned.
TP=4

mkdir -p tools/convert_fp8/logs
SERVER_LOG=tools/convert_fp8/logs/sglang_server_$(date +%Y%m%d_%H%M%S)_$$.log

python -m sglang.launch_server \
  --model-path "$MODEL" \
  --quantization fp8 \
  --tp "$TP" \
  --port "$PORT" \
  --trust-remote-code \
  > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
echo "SERVER_PID=$SERVER_PID"
echo "SERVER_LOG=$SERVER_LOG"
