#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
# One compute thread per rank; GPU does the heavy lifting, avoid CPU oversubscription.
export OMP_NUM_THREADS=1
DEVICE="${DEVICE:-cuda}"

cp /etc/mpi/hostfile /root/hostfile
mpirun -v --allow-run-as-root \
  --bind-to none --map-by slot --hostfile /root/hostfile \
  --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
  python3 -u "tools/dsv4_quant_and_dequant/compare_hf_ckpts.py" \
  --a test_convert_jingt/fp8_2_fp4_2_fp8 \
  --b /mnt/geminigmceph/user_yyyuuuzhang/code/gcore-dev-0710/ckpt_miniprogram_grpo_sglang/hf_fp4_fp8/150/ \
  --a-name converted --b-name published_official --device cuda
