#!/usr/bin/env bash
# Run every test file individually with ray stop/init between each.
# Usage: cd /work/wepsdl/gcore-dev && bash tests/each_test.sh 2>&1 | tee /tmp/each_test_$(date +%Y%m%d_%H%M%S).log
set -o pipefail

RCDIR="/work/wepsdl"
export PYTHONPATH="${PYTHONPATH:-}"
export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/gcore-dev/tests/test_gpatch_v4:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"

TIMEOUT=1800
PASS=0
FAIL=0
SKIP=0
ERRORS=()

run_one() {
    local test_args="$1"
    echo ""
    echo "================================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START: $test_args"
    echo "================================================================"

    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh

    pytest -v -s --timeout=$TIMEOUT $test_args
    local rc=$?

    if [ $rc -eq 0 ]; then
        echo "[RESULT] $test_args => PASSED"
        PASS=$((PASS + 1))
    elif [ $rc -eq 5 ]; then
        # pytest exit code 5 = no tests collected (all skipped)
        echo "[RESULT] $test_args => SKIPPED (no tests collected)"
        SKIP=$((SKIP + 1))
    else
        echo "[RESULT] $test_args => FAILED (exit=$rc)"
        FAIL=$((FAIL + 1))
        ERRORS+=("$test_args")
    fi
}

# ---- Already PASSED (26/59), skip for now ----
# run_one tests/test_gfused/test_flash_attn_triton.py
# run_one tests/test_gfused/test_gated_delta_net.py
# run_one tests/test_gpatch_v4/test_actor_cleanup_sglang.py
# run_one tests/test_gpatch_v4/test_advantage_clip_bounds.py
# run_one tests/test_gpatch_v4/test_agentic_rl.py
# run_one tests/test_gpatch_v4/test_allocation_from_config.py
# run_one tests/test_gpatch_v4/test_async_rollout_sliding_prefetch.py
# run_one tests/test_gpatch_v4/test_bt_rm.py
# run_one tests/test_gpatch_v4/test_ckpt.py
# run_one tests/test_gpatch_v4/test_ckpt_interval_naming.py
# run_one tests/test_gpatch_v4/test_custom_config.py
# run_one tests/test_gpatch_v4/test_data_flow.py
# run_one tests/test_gpatch_v4/test_device_backend_priv.py
# run_one tests/test_gpatch_v4/test_dp_balance.py
# run_one tests/test_gpatch_v4/test_expected_nnodes.py
# run_one tests/test_gpatch_v4/test_external_reward.py
# run_one tests/test_gpatch_v4/test_failure_recovery.py
# run_one tests/test_gpatch_v4/test_fsdp.py
# run_one tests/test_gpatch_v4/test_gen_rm.py
# run_one tests/test_gpatch_v4/test_hetero_gen_rm.py
# run_one tests/test_gpatch_v4/test_hf_metadata_cache.py
# run_one tests/test_gpatch_v4/test_hydra.py
# run_one tests/test_gpatch_v4/test_infer_only_mode.py
# run_one tests/test_gpatch_v4/test_langgraph.py
# run_one tests/test_gpatch_v4/test_llm_actor.py
# run_one tests/test_gpatch_v4/test_logging_utils.py

# ---- Re-test: test_math_r3.py 3 remaining failed sglang sub-tests ----
run_one "tests/test_gpatch_v4/test_math_r3.py -k test_filter_sampling_pre_sglang"
run_one "tests/test_gpatch_v4/test_math_r3.py -k test_filter_sampling_post_sglang"
run_one "tests/test_gpatch_v4/test_math_r3.py -k test_async_rollout_grpo_two_turns_rollout_sglang"

# ---- Summary ----
echo ""
echo "================================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUMMARY"
echo "================================================================"
echo "  PASSED : $PASS"
echo "  FAILED : $FAIL"
echo "  SKIPPED: $SKIP"
if [ ${#ERRORS[@]} -gt 0 ]; then
    echo ""
    echo "  Failed test files:"
    for f in "${ERRORS[@]}"; do
        echo "    - $f"
    done
fi
echo "================================================================"
