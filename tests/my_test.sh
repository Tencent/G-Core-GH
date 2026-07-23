#!/bin/bash
# 非 DeepSeek-V4 回归测试脚本
# set -e

RCDIR="../"
export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/gcore-dev/tests/test_gpatch_v4:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"

# ============================================================
# 多 GPU 集群测试（需要 Ray）
# ============================================================
echo "=== 初始化 Ray 集群 ==="
source tests/test_gpatch_v4/mpirun-stop-ray.sh
source tests/test_gpatch_v4/mpirun-init-ray.sh

# 跑非 DSV4 用例（排除 deepseek_v4 / dsv4）
pytest -v -s --timeout=7200 \
  tests/test_gfused/ \
  tests/test_gpatch_v4/ \
  --ignore-glob='*deepseek_v4*' \
  --ignore-glob='*dsv4*' \
  --ignore=tests/test_gpatch_v4/test_sgl_qwen36moe_release_resume_v4.py \
  --ignore=tests/test_gpatch_v4/test_gemini_self_heal.py

# DSV4 全量（备用）
# pytest -v -s --timeout=7200 \
#   tests/test_gfused/test_deepseek_v4_*.py \
#   tests/test_gpatch_v4/test_deepseek_v4_*.py

# pytest -v -s tests/test_gfused/test_sp_overlap.py
# pytest -v -s tests/test_gpatch_v4/test_ray_and_nsys.py
# pytest -v -s tests/test_gpatch_v4/test_fsdp2_fp8_all_gather.py
