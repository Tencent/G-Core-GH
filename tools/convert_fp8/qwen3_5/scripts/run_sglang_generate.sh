#!/bin/bash
set -euo pipefail

cd /work/wepsdl/gcore-dev

MODEL=/work/wepsdl/gcore-dev/Qwen3.6-35B-A3B-FP8-official-skip
PORT=30000
# Qwen3.6 MoE expert hidden size is 512; SGLang splits gate/up by TP before
# FP8 block setup, so TP=8 gives 64 which is not 128-aligned.
TP=4

for i in $(seq 1 120); do
  if curl -fs "http://127.0.0.1:${PORT}/health" >/dev/null; then
    echo "SGLang server ready"
    break
  fi
  sleep 5
done

python tools/convert_fp8/qwen3_5/tests/test_fp8_sglang_generate.py \
  --server-url "http://127.0.0.1:${PORT}" \
  --max-tokens 128 \
