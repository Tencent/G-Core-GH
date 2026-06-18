#!/bin/bash
# DeepSeek V4 全量回归测试脚本
# 分三类：纯CPU测试 → 单GPU测试 → 多GPU集群测试(Ray)
# set -e

RCDIR="../"
export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/gcore-dev/tests/test_gpatch_v4:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"

# ============================================================
# 第三部分：多 GPU 集群测试（需要 Ray）
# ============================================================
echo "=== 初始化 Ray 集群 ==="
source tests/test_gpatch_v4/mpirun-stop-ray.sh
source tests/test_gpatch_v4/mpirun-init-ray.sh

# pytest -v -s --timeout=7200 tests/test_gfused/test_deepseek_v4_kernel_ops.py
# pytest -v -s --timeout=2400 tests/test_gfused/test_deepseek_v4_ep_cp_thd.py::TestEpCpThd::test_pack_runs_eager_topk
pytest -v -s --timeout=2400 tests/test_gfused/test_deepseek_v4_ep_cp_thd.py::TestEpCpThd::test_pack_runs_long
# pytest -v -s --timeout=2400 tests/test_gfused/test_deepseek_v4_ep_cp_thd.py::TestEpCpThd::test_pack_runs_fused
# pytest -v -s --timeout=7200 tests/test_gfused/test_deepseek_v4_ep_cp.py::TestFsdpVsEpCp::test_cp_memory_scaling
# pytest -v -s --timeout=7200 tests/test_gfused/test_dsv4_mtp_passthrough.py
# pytest -v -s --timeout=7200 tests/test_gfused/test_dsv4_mtp_cp_roll.py
# pytest -v -s --timeout=7200 tests/test_gfused/test_deepseek_v4_mtp_ep_parity.py
# pytest -v -s --timeout=7200 tests/test_gfused/test_deepseek_v4_mtp_smoke.py
# pytest -v -s --timeout=7200 tests/test_gfused/test_deepseek_v4_ep.py::TestFsdpVsEp::test_fsdp_vs_ep_deepep
# pytest -v -s --timeout=7200 tests/test_gfused/test_deepseek_v4_ep_cp.py::TestFsdpVsEpCp::test_fsdp_vs_ep_cp_s512
# pytest -v -s --timeout=7200 tests/test_gfused/test_myfa_hca_fwd.py::test_bench

# pytest -v -s --timeout=7200 tests/test_gfused/test_myfa_varlen.py
# pytest -v -s --timeout=7200 tests/test_gfused/test_myfa_varlen.py::test_fwd_bwd
# pytest -v -s --timeout=7200 tests/test_gfused/test_myfa_varlen.py::test_bench_fwd_throughput
# pytest -v -s --timeout=7200 tests/test_gfused/test_myfa_varlen.py::test_bench_bwd_throughput

pytest -v -s --timeout=7200 tests/test_gfused/test_myfa_mask.py

echo "=== 清理 Ray 集群 ==="
source tests/test_gpatch_v4/mpirun-stop-ray.sh