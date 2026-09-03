#!/usr/bin/env bash
# DeepSeek-V4 multi-teacher OPD one-step regression test.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
MCORE_PATH="${MEGATRON_LM_PATH:-${MEGATRON_LM_ROOT:-/root/Megatron-LM}}"
MBRIDGE_PATH="${MBRIDGE_PATH:-${MBRIDGE_ROOT:-/root/mbridge}}"
MEGATRON_BRIDGE_PATH="${MEGATRON_BRIDGE_PATH:-${MEGATRON_BRIDGE_ROOT:-/root/Megatron-Bridge}}"
MODEL_PATH="/mnt/wfs/mmhuizhouwfssz/project_luban_infra/luban_infra/model_factory/lucasbai_DeepSeek-V4-Flash-FP8"

cd "$REPO_ROOT"
echo "REPO_ROOT: $REPO_ROOT"

for required_path in \
  "$MCORE_PATH" \
  "$MBRIDGE_PATH" \
  "$MEGATRON_BRIDGE_PATH/src" \
  "$MODEL_PATH/config.json" \
  "$MODEL_PATH/model.safetensors.index.json" \
  "$MODEL_PATH/tokenizer.json" \
  "$MODEL_PATH/tokenizer_config.json" \
  "tasks/agentic_rl/sokoban_rl/scripts/mpirun-init-ray.sh"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Missing required path: $required_path" >&2
    exit 1
  fi
done

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/tests:$REPO_ROOT/tests/test_gpatch_v4:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:${PYTHONPATH:-}"

# Match the environment used by the DeepSeek-V4 MOPD launcher.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export RAY_DEDUP_LOGS=0
export PYTHONUNBUFFERED=1
export VLLM_LOG_STATS_INTERVAL="${VLLM_LOG_STATS_INTERVAL:-3}"
# topk_v2 takes the incompatible NVCC JIT path seen in the failed worker log.
export SGLANG_OPT_USE_TOPK_V2=0
export VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY:-0}"
if [[ "$VLLM_ENABLE_CUDA_COMPATIBILITY" == "1" ]]; then
  export VLLM_CUDA_COMPATIBILITY_PATH="/usr/local/cuda-12.8/compat"
else
  export VLLM_CUDA_COMPATIBILITY_PATH="${VLLM_CUDA_COMPATIBILITY_PATH:-}"
fi
export GPATCH_EXTRA_PROPAGATE_ENV="VLLM_LOG_STATS_INTERVAL,VLLM_ENABLE_CUDA_COMPATIBILITY,VLLM_CUDA_COMPATIBILITY_PATH,SGLANG_OPT_USE_TOPK_V2"

# The YAML requests 28 eight-GPU nodes. Both names are set because the
# reference Ray launcher and the test cleanup script use different variables.
export GCORE_GPU="${GCORE_GPU:-28}"
export GCORE_NNODES="${GCORE_NNODES:-$GCORE_GPU}"
if (( GCORE_GPU != 28 || GCORE_NNODES != 28 )); then
  echo "This test requires exactly 28 nodes; got GCORE_GPU=$GCORE_GPU and GCORE_NNODES=$GCORE_NNODES" >&2
  exit 1
fi
available_nodes="$(wc -l < /etc/mpi/hostfile)"
if (( available_nodes < 28 )); then
  echo "This test requires 28 nodes, but /etc/mpi/hostfile contains $available_nodes" >&2
  exit 1
fi

# Do not silently kill an expensive run that is still active.
active_training="$(pgrep -af 't[r]ain_mopd.py|p[y]test.*test_frozen_lake_mopd.py' || true)"
if [[ -n "$active_training" && "${FORCE_RAY_RESTART:-0}" != "1" ]]; then
  echo "Refusing to reset Ray while another MOPD/test process is active:" >&2
  echo "$active_training" >&2
  echo "Wait for it to finish, or set FORCE_RAY_RESTART=1 to replace it intentionally." >&2
  exit 1
fi

# The test disables WandB, but the shared MPI launcher exports these variables.
export WANDB_API_KEY=
export WANDB_BASE_URL=

echo "=== Verifying Hydra config and 28-node topology ==="
pytest -q -p no:cacheprovider \
  tests/test_gpatch_v4/test_frozen_lake_mopd.py \
  -k test_config_routes_each_environment_to_a_teacher

# ============================================================
# Multi-node environment preparation and Ray startup.
#
# This follows the production MOPD launcher. It cleans stale Ray/SGLang
# processes, applies the tilelang CUDA runtime fix, prepares host memory,
# installs agentic environment dependencies, and propagates the DSV4 flags.
# ============================================================
echo "=== Preparing 28-node Ray cluster ==="
# Stop stale raylets on every node first. The production launcher only stops
# the local raylet, which is insufficient when rerunning in one allocation.
source tests/test_gpatch_v4/mpirun-stop-ray.sh
source tasks/agentic_rl/sokoban_rl/scripts/mpirun-init-ray.sh

echo "=== Verifying Ray cluster and DSV4 environment on every node ==="
python3 - <<'PY'
import os

import ray
from gpatch_v4.orches.utils import build_actor_env_vars
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

EXPECTED_NODES = 28
EXPECTED_GPUS = 224
MODEL_PATH = (
    "/mnt/wfs/mmhuizhouwfssz/project_luban_infra/luban_infra/model_factory/"
    "lucasbai_DeepSeek-V4-Flash-FP8"
)
REQUIRED_PATHS = (
    "/root/Megatron-LM",
    "/root/mbridge",
    "/root/Megatron-Bridge/src",
    "/root/sglang",
    f"{MODEL_PATH}/config.json",
    f"{MODEL_PATH}/model.safetensors.index.json",
    f"{MODEL_PATH}/tokenizer.json",
    f"{MODEL_PATH}/tokenizer_config.json",
)

ray.init(address="auto")
try:
    nodes = [node for node in ray.nodes() if node["Alive"]]
    resources = ray.cluster_resources()
    assert len(nodes) == EXPECTED_NODES, (
        f"expected {EXPECTED_NODES} live Ray nodes, got {len(nodes)}"
    )
    assert int(resources.get("GPU", 0)) == EXPECTED_GPUS, (
        f"expected {EXPECTED_GPUS} Ray GPUs, got {resources.get('GPU', 0)}"
    )
    actor_env = build_actor_env_vars()
    assert actor_env.get("SGLANG_OPT_USE_TOPK_V2") == "0", (
        "gpatch actor runtime_env does not contain SGLANG_OPT_USE_TOPK_V2=0"
    )

    @ray.remote(num_cpus=0, runtime_env={"env_vars": actor_env})
    def check_node():
        missing = [path for path in REQUIRED_PATHS if not os.path.exists(path)]
        return {
            "host": os.uname().nodename,
            "missing": missing,
            "topk_v2": os.environ.get("SGLANG_OPT_USE_TOPK_V2"),
        }

    checks = ray.get([
        check_node.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node["NodeID"],
                soft=False,
            )
        ).remote()
        for node in nodes
    ])
    failures = [
        check
        for check in checks
        if check["missing"] or check["topk_v2"] != "0"
    ]
    assert not failures, f"Ray node preflight failed: {failures}"
    print(
        f"Ray preflight passed: {len(nodes)} nodes, "
        f"{int(resources['GPU'])} GPUs, SGLANG_OPT_USE_TOPK_V2=0",
        flush=True,
    )
finally:
    ray.shutdown()
PY

pytest -v -s --timeout=7200 \
  tests/test_gpatch_v4/test_frozen_lake_mopd.py \
  -k test_train_one_step_sglang
