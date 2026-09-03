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

# ============================================================
# 2026-07-23 全量 fail 子集（默认）
# 来源: 28 failed / 838 passed / 29 skipped in 6h04m
# 不含 myfa_hca_fwd（代码未写完，已 ignore）
# 不含 ceph 孤儿（dynamic_cp / myfa_welm_45_moe / qwen3_5_moe_ep；应删除）
# ============================================================
# pytest -v -s --timeout=7200 \
#   tests/test_gfused/test_sp_overlap.py::Test1::test_ulysses_e2e_timeline \
#   tests/test_gfused/test_tmp.py::TestTmp1::test_tmp_1 \
#   tests/test_gpatch_v4/test_actor_cleanup_sglang.py::ActorCleanupSglangTest::test_gen_rm_actor_cleanup_releases_gpu_and_kills_sglang \
#   tests/test_gpatch_v4/test_deep_gemm_grouped_gemm_fp8.py::test_deep_gemm_grouped_gemm_fp8_correctness \
#   tests/test_gpatch_v4/test_hf_metadata_cache.py::test_mbridge_save_hf_uses_existing_safetensor_io \
#   tests/test_gpatch_v4/test_qwen3_6_moe_text_only_sft.py::TestQwen36MoETextOnlyFwd::test_fwd_bwd \
#   tests/test_gpatch_v4/test_qwen3_6_moe_text_only_sft.py::TestQwen36MoETextOnlyFwd::test_fwd_bwd_with_cp \
#   tests/test_gpatch_v4/test_qwen3_6_moe_text_only_sft.py::TestHFvsMegatron::test_hf_vs_megatron \
#   tests/test_gpatch_v4/test_qwen3_6_moe_text_only_sft.py::TestHFvsMegatron::test_hf_vs_megatron_with_cp \
#   tests/test_gpatch_v4/test_ray_and_nsys.py::test_ray_nsys_profiles_matmul_nccl_barrier \
#   tests/test_gpatch_v4/test_router_replay_r3.py::RouterReplayR3Test::test_r3_sglang \
#   tests/test_gpatch_v4/test_sampler_get_load.py::SamplerGetLoadTest::test_get_load_idle_and_under_generation \
#   tests/test_gpatch_v4/test_sampler_get_load.py::SamplerGetLoadTest::test_train_async_rollout_load_aware \
#   tests/test_gpatch_v4/test_sgl_qwen3_vs_qwen36.py::Qwen3vsQwen36ThroughputTest::test_throughput_comparison \
#   tests/test_gpatch_v4/test_sgl_welm_v45_router_replay.py::test_sglang_generation_e2e

# ============================================================
# 全量（备用）
# ============================================================
# pytest -v -s --timeout=7200 \
#   tests/test_gfused/ \
#   tests/test_gpatch_v4/ \
#   --ignore-glob='*deepseek_v4*' \
#   --ignore-glob='*dsv4*' \
#   --ignore=tests/test_gpatch_v4/test_sgl_qwen36moe_release_resume_v4.py \
#   --ignore=tests/test_gpatch_v4/test_gemini_self_heal.py \
#   --ignore=tests/test_gfused/test_myfa_hca_fwd.py

# DSV4 全量（备用）
# pytest -v -s --timeout=1800 \
#   tests/test_gfused/test_deepseek_v4_*.py \
#   tests/test_gpatch_v4/test_deepseek_v4_*.py

# pytest -v -s --timeout=7200 tests/test_gfused/test_deepseek_v4_perf.py::TestDeepseekV4Perf::test_hp_fused_forward_timeline
# pytest -v -s --timeout=7200 tests/test_gfused/test_fa2_fa3_throughput.py
# pytest -v -s --timeout=7200 tests/test_gfused/test_sp_overlap.py
# pytest -v -s --timeout=7200 tests/test_gfused/test_deepseek_tilekernels_quant.py::TestTileKernelsVsEagerFp8Block
# pytest -v -s --timeout=7200 tests/test_gfused/test_te_gemm_fp8.py::TestTeGroupedGemmFp8CustomOp tests/test_gpatch_v4/test_fsdp2_fp8_all_gather.py
# pytest -v -s --timeout=7200 tests/test_gpatch_v4/test_fsdp2_fp8_all_gather.py

# ============================================================
# myfa_varlen_sw_sinks（lru_cache / compile kwargs 透传）
# ============================================================
# pytest -v -s --timeout=7200 \
#   tests/test_gfused/test_myfa_varlen_sw_sinks.py::test_fwd_bwd \
#   tests/test_gfused/test_myfa_varlen_sw_sinks.py::test_bench_fwd \
#   tests/test_gfused/test_myfa_varlen_sw_sinks.py::test_bench_bwd
# 同族（非 varlen）备用：
# pytest -v -s --timeout=7200 \
#   tests/test_gfused/test_myfa_sw_sinks.py::test_fwd_bwd \
#   tests/test_gfused/test_myfa_sw_sinks.py::test_fwd_sft_cp_shape_q_offsets_smoke \
#   tests/test_gfused/test_myfa_sw_sinks.py::test_bench_fwd \
#   tests/test_gfused/test_myfa_sw_sinks.py::test_bench_bwd \
#   tests/test_gfused/test_myfa_sw_sinks.py::test_bench_fwd_sft_cp_shape_q_offsets

# pytest -v -s --timeout=7200 tests/test_gfused/test_my_gemm_fp8.py
# pytest -v -s --timeout=7200 tests/test_gfused/test_my_gemm_fp8.py::test_perf_gemm_fp8_blk_quant
# pytest -v -s --timeout=7200 tests/test_gfused/test_my_gemm_fp8.py::test_perf_my_vs_te
# pytest -v -s --timeout=7200 tests/test_gfused/test_my_gemm_fp8.py::test_gemm_fp8_blk_quant
# pytest -v -s --timeout=7200 tests/test_gfused/test_deepseek_v4_indexer.py::test_indexer_fwd_fp8_matches_fake_quant_bf16
# pytest -v -s --timeout=7200 tests/test_gfused/test_deepseek_v4_indexer.py::test_perf_indexer_fwd_fp8

# ============================================================
# DSV4 modeling: config.fp8 + fused indexer（per_token_cast → FP8 fwd）
# ============================================================
pytest -v -s --timeout=7200 \
  tests/test_gfused/test_deepseek_v4_ep_cp_thd.py::TestEpCpThd::test_pack_runs_fused_fp8 \
  tests/test_gfused/test_deepseek_v4_ep_cp_thd.py::TestEpCpThd::test_hp_fused_smoke
