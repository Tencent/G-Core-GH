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

# pytest -v -s --timeout=1800 tests/test_gfused/test_deepseek_v4_allgather_perf.py::DeepSeekV4AllGatherPerfTest::test_allgather_avg_time
# pytest -v -s --timeout=1800 tests/test_gfused/test_deepseek_v4_allgather_perf.py::DeepSeekV4AllGatherPerfTest::test_indexer_causal_head_tail
# pytest -v -s --timeout=2400 tests/test_gfused/test_deepseek_v4_ep_cp_thd.py::TestEpCpThd::test_bench_long
# pytest -v -s --timeout=2400 tests/test_gfused/test_deepseek_v4_ep_cp_thd.py::TestEpCpThd::test_cp_balance_loss_equivalence
# pytest -v -s --timeout=2400 tests/test_gfused/test_deepseek_v4_ep_cp_thd.py::TestEpCpThd::test_pack_runs_eager
# pytest -v -s --timeout=2400 tests/test_gfused/test_deepseek_v4_ep_cp_thd.py::TestEpCpThd::test_pack_runs_eager_topk
# pytest -v -s --timeout=2400 tests/test_gfused/test_deepseek_v4_ep_cp_thd.py::TestEpCpThd::test_pack_runs_fused
# pytest -v -s --timeout=2400 tests/test_gfused/test_deepseek_v4_ep_cp_thd.py::TestEpCpThd::test_pack_runs_fused_fp8
# pytest -v -s --timeout=2400 tests/test_gfused/test_deepseek_v4_qat.py
# pytest -v -s --timeout=1800 tests/test_gfused/test_nccl_bench.py::AllToAllOverheadTest::test_bf16_all_gather_1m_1024
# pytest -v -s --timeout=7200 tests/test_gfused/test_deepseek_v4_ep.py::TestFsdpVsEp::test_fsdp_vs_ep_deepep
# pytest -v -s --timeout=7200 tests/test_gfused/test_deepseek_v4_ep_cp.py::TestFsdpVsEpCp::test_cp_memory_scaling
# pytest -v -s --timeout=7200 tests/test_gfused/test_deepseek_v4_ep_cp.py::TestFsdpVsEpCp::test_fsdp_vs_ep_cp_s512
# pytest -v -s --timeout=7200 tests/test_gfused/test_deepseek_v4_kernel_ops.py
# pytest -v -s --timeout=7200 tests/test_gfused/test_deepseek_v4_mtp_ep_parity.py
# pytest -v -s --timeout=7200 tests/test_gfused/test_deepseek_v4_mtp_smoke.py
# pytest -v -s --timeout=7200 tests/test_gfused/test_deepseek_v4_mtp_smoke.py::TestDeepseekV4MtpSmoke::test_mtp_bshd_vs_thd
# pytest -v -s --timeout=7200 tests/test_gfused/test_deepseek_v4_mtp_smoke.py::TestDeepseekV4MtpSmoke::test_mtp_thd_vs_thd_cp
# pytest -v -s --timeout=7200 tests/test_gfused/test_dsv4_mtp_cp_roll.py
# pytest -v -s --timeout=7200 tests/test_gfused/test_dsv4_mtp_passthrough.py
# pytest -v -s --timeout=7200 tests/test_gfused/test_myfa_hca_fwd.py::test_bench
# pytest -v -s --timeout=7200 tests/test_gfused/test_myfa_varlen.py
# pytest -v -s --timeout=7200 tests/test_gfused/test_myfa_varlen.py::test_bench_fwd_throughput
# pytest -v -s --timeout=7200 tests/test_gfused/test_myfa_varlen.py::test_fwd_bwd
# pytest -v -s --timeout=7200 tests/test_gfused/test_myfa_varlen_sw_sinks.py::test_bench_bwd
# pytest -v -s --timeout=7200 tests/test_gfused/test_myfa_varlen_sw_sinks.py::test_bench_fwd
# pytest -v -s --timeout=7200 tests/test_gpatch_v4/test_deepseek_v4_sft_thd.py::TestDsv4SftThd::test_thd_mtp_vs_bshd_mtp
# pytest -v -s tests/test_gfused/test_te_gemm_fp8.py
# pytest -v -s tests/test_gfused/test_te_gemm_fp8.py::TestTeGroupedGemmFp8CustomOp
# pytest -v -s tests/test_gpatch_v4/test_fsdp2_balance_loss.py
# pytest -v -s --timeout=2400 tests/test_gfused/test_deepseek_v4_ep_cp_thd.py::TestEpCpThd::test_hp_fused_smoke
# pytest -v -s --timeout=1800 tests/test_gfused/test_deepseek_v4_perf.py::TestDeepseekV4Perf::test_hp_fused_forward_timeline

# pytest -v -s --timeout=1800 tests/test_gfused/test_deepseek_v4_fp4_qat.py
# pytest -v -s --timeout=2400 tests/test_gpatch_v4/test_fp4_qat_config.py
pytest -v -s --timeout=2400 tests/test_gfused/test_deepseek_v4_qat.py
# pytest -v -s --timeout=2400 tests/test_gfused/test_deepseek_v4_ep_cp_thd.py::TestEpCpThd::test_pack_runs_long