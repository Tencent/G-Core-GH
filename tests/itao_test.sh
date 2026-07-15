#!/bin/bash
# set -e

RCDIR="../"
export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/gcore-dev/tests/test_gpatch_v4:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"

# ============================================================
# 第三部分：多 GPU 集群测试（需要 Ray）
# ============================================================
# echo "=== 初始化 Ray 集群 ==="
export GCORE_GPU=2
source tests/test_gpatch_v4/mpirun-stop-ray.sh
source tests/test_gpatch_v4/mpirun-init-ray.sh
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SGLANG_OPT_USE_TOPK_V2=0
# pytest -v -s --timeout=1800 tests/test_gpatch_v4/test_cp_mappings.py
# pytest -v -s --timeout=1800 tests/test_gfused/test_router_replay.py
# pytest -v -s --timeout=1800 tests/test_gpatch_v4/test_opd_topk_linear_ce_fused.py
# pytest -v -s --timeout=1800 tests/test_gpatch_v4/test_advantage_clip_bounds.py
# pytest -v -s --timeout=1800 tests/test_gpatch_v4/test_update_weight_factory.py
pytest -v -s --timeout=1800 tests/test_gpatch_v4/test_sgl_dsv4.py

echo "=== 清理 Ray 集群 ==="
source tests/test_gpatch_v4/mpirun-stop-ray.sh
