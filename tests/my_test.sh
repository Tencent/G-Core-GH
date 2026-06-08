#!/bin/bash
# DeepSeek V4 全量回归测试脚本
# 分三类：纯CPU测试 → 单GPU测试 → 多GPU集群测试(Ray)
# set -e

RCDIR="../"
export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/gcore-dev/tests/test_gpatch_v4:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"

# ============================================================
# 第一部分：纯 CPU 测试（快速，无 GPU 依赖）
# ============================================================
# echo "=== [1/20] test_dsv4_mtp_cp_roll ==="
# pytest -v -s --timeout=300 tests/test_gfused/test_dsv4_mtp_cp_roll.py
# 
# echo "=== [2/20] test_router_replay ==="
# pytest -v -s --timeout=300 tests/test_gfused/test_router_replay.py
# 
# echo "=== [3/20] test_build_cp_causal_mask ==="
# pytest -v -s --timeout=300 tests/test_gfused/test_build_cp_causal_mask.py
# 
# echo "=== [4/20] test_convert_deepseek_v4_fp4_to_bf16 ==="
# pytest -v -s --timeout=1800 tests/test_tools/test_convert_deepseek_v4_fp4_to_bf16.py
# 
# # ============================================================
# # 第二部分：单 GPU 测试（需要 GPU，不需要 Ray）
# # ============================================================
# echo "=== [5/20] test_dsv4_rename_table ==="
# pytest -v -s --timeout=600 tests/test_gfused/test_dsv4_rename_table.py
# 
# echo "=== [6/20] test_dsv4_chat_template ==="
# pytest -v -s --timeout=600 tests/test_gfused/test_dsv4_chat_template.py
# 
# echo "=== [7/20] test_dsv4_classify_audit ==="
# pytest -v -s --timeout=600 tests/test_gfused/test_dsv4_classify_audit.py
# 
# echo "=== [8/20] test_dsv4_mtp_passthrough ==="
# pytest -v -s --timeout=600 tests/test_gfused/test_dsv4_mtp_passthrough.py
# 
# echo "=== [9/20] test_deepseek_v4_hf ==="
# pytest -v -s --timeout=1800 tests/test_gfused/test_deepseek_v4_hf.py
# 
# echo "=== [10/20] test_deepseek_v4_thd ==="
# pytest -v -s --timeout=1800 tests/test_gfused/test_deepseek_v4_thd.py
# 
# echo "=== [11/20] test_deepseek_v4_kernel_ops ==="
# # pytest -v -s --timeout=1800 tests/test_gfused/test_deepseek_v4_kernel_ops.py
# 
# echo "=== [12/20] test_deepseek_v4_pack_seq_agnostic ==="
# pytest -v -s --timeout=1800 tests/test_gfused/test_deepseek_v4_pack_seq_agnostic.py
# 
# echo "=== [13/20] test_fp_quantize (DeepSeek V4 related) ==="
# pytest -v -s --timeout=1200 tests/test_gfused/test_fp_quantize.py::TestDeepseekCheckpointQuantDequant
# pytest -v -s --timeout=1200 tests/test_gfused/test_fp_quantize.py::TestQuantFp8AgainstOcpMxfp8::test_dsv4_distribution

# ============================================================
# 第三部分：多 GPU 集群测试（需要 Ray）
# ============================================================
echo "=== 初始化 Ray 集群 ==="
source tests/test_gpatch_v4/mpirun-stop-ray.sh
source tests/test_gpatch_v4/mpirun-init-ray.sh

# echo "=== [14/20] test_deepseek_v4_ep ==="
# pytest -v -s --timeout=2400 tests/test_gfused/test_deepseek_v4_ep.py
# 
# echo "=== [15/20] test_deepseek_v4_ep_cp ==="
# pytest -v -s --timeout=2400 tests/test_gfused/test_deepseek_v4_ep_cp.py

# echo "=== [16/20] test_deepseek_v4_ep_cp_thd ==="
# pytest -v -s --timeout=2400 tests/test_gfused/test_deepseek_v4_ep_cp_thd.py
# 
# echo "=== [17/20] test_deepseek_v4_mtp_smoke ==="
# pytest -v -s --timeout=1200 tests/test_gfused/test_deepseek_v4_mtp_smoke.py
# 
# echo "=== [18/20] test_deepseek_v4_mtp_ep_parity ==="
# pytest -v -s --timeout=2400 tests/test_gfused/test_deepseek_v4_mtp_ep_parity.py
 
echo "=== [19/20] test_deepseek_v4_save_load ==="
pytest -v -s --timeout=3600 tests/test_gfused/test_deepseek_v4_save_load.py

# echo "=== [20/20] test_deepseek_v4_keep_fp32 ==="
# pytest -v -s --timeout=1800 tests/test_gfused/test_deepseek_v4_keep_fp32.py

echo "=== 清理 Ray 集群 ==="
source tests/test_gpatch_v4/mpirun-stop-ray.sh

echo ""
echo "=== DeepSeek V4 全量回归测试完成 ==="
