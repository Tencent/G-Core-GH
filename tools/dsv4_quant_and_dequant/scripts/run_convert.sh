#!/usr/bin/env bash
# Convert DeepSeek-V4-Flash between official and SGLang FP8 layouts.
# Multi-GPU via mpirun: one rank per GPU, size-balanced shard split.

set -euo pipefail

# official2sgl|sgl2official
DIRECTION="official2sgl"
SRC="test_convert_jingt/fp8_2_fp4"
DST="test_convert_jingt/fp8_2_fp4_2_fp8"

# DIRECTION="sgl2official"
# SRC="/mnt/geminigmceph/user_yyyuuuzhang/code/gcore-dev-0710/ckpt_miniprogram_grpo_sglang/hf/150/"
# DST="test_convert_jingt/fp8_2_fp4"


export PYTHONPATH="$PWD:${PYTHONPATH:-}"
# One compute thread per rank; GPU does the heavy lifting, avoid CPU oversubscription.
export OMP_NUM_THREADS=1
DEVICE="${DEVICE:-cuda}"

cp /etc/mpi/hostfile /root/hostfile
mpirun -v --allow-run-as-root \
  --bind-to none --map-by slot --hostfile /root/hostfile \
  --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
  python3 -u "tools/dsv4_quant_and_dequant/convert.py" \
  --direction "${DIRECTION}" \
  --src "${SRC}" \
  --dst "${DST}" \
  --device "${DEVICE}"
