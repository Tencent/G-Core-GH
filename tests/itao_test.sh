#!/bin/bash
# set -e

RCDIR="../"
export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/gcore-dev/tests/test_gpatch_v4:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"

# ============================================================
# 第三部分：多 GPU 集群测试（需要 Ray）
# ============================================================
# echo "=== 初始化 Ray 集群 ==="
export GCORE_NNODES=4
source tests/test_gpatch_v4/mpirun-stop-ray.sh
source tests/test_gpatch_v4/mpirun-init-ray.sh
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SGLANG_OPT_USE_TOPK_V2=0

# pytest -v -s --timeout=1800 tests/test_gpatch_v4/test_loss_registry.py
# pytest -v -s --timeout=1800 tests/test_gpatch_v4/test_loss_backend_smoke.py
# pytest -v -s --timeout=1800 tests/test_gpatch_v4/test_fsdp2_linear_ce.py
# pytest -v -s --timeout=1800 tests/test_gpatch_v4/test_steer_smart_pad.py
# pytest -v -s --timeout=1800 tests/test_gpatch_v4/test_steer_loss.py
# pytest -v -s --timeout=1800 tests/test_gfused/test_deepseek_v4_kernel_ops.py
# pytest -v -s --timeout=1800 tests/test_gpatch_v4/test_external_reward.py::AsyncEvalExternalRewardTest
# pytest -v -s --timeout=1800 tests/test_gpatch_v4/test_quantize_kernels.py
# pytest -v -s --timeout=1800 tests/test_gpatch_v4/test_sglang_weight_update_dsv4.py
# pytest -v -s --timeout=1800 tests/test_gpatch_v4/test_loss_registry.py
# pytest -v -s --timeout=1800 tests/test_gpatch_v4/test_mlite_batch_bridge.py
# pytest -v -s --timeout=1800 tests/test_gpatch_v4/test_mlite_engine_contract.py
# pytest -v -s --timeout=1800 tests/test_gpatch_v4/test_mlite_loss_scaling.py
pytest -v -s --timeout=1800 tests/test_gpatch_v4/test_mlite_engine_contract.py
pytest -v -s --timeout=1800 tests/test_gpatch_v4/test_welm_mlite_batch_bridge.py
pytest -v -s --timeout=1800 tests/test_gpatch_v4/test_welm_mlite_engine_contract.py
pytest -v -s --timeout=1800 tests/test_gpatch_v4/test_welm_v45_myfa_packed_cp.py
pytest -v -s --timeout=1800 tests/test_gfused/test_myfa_varlen_sw_sinks.py





# echo "=== 清理 Ray 集群 ==="
source tests/test_gpatch_v4/mpirun-stop-ray.sh
