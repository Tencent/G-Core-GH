# Shared verl env for the alignment runs: PYTHONPATH (with the flashinfer /
# determinism bootstrap), deterministic-compute env vars, wandb, and a multi-node
# Ray bootstrap helper. Source this from a per-algorithm run.sh.
#
# _bootstrap holds a sitecustomize.py that (1) redirects flashinfer's libcudart_stub
# load to the real libcudart and (2) enables torch determinism + unloads FA3 (gated
# on GCORE_VERL_DETERMINISTIC). It must be FIRST on PYTHONPATH so Python auto-imports
# it at startup in the driver and every Ray worker before sglang imports flashinfer.
COMMON_VERL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_PATH="$COMMON_VERL_DIR/_bootstrap"

# verl repo (on the shared distributed FS, identical locally and on the test node)
VERL_PATH=${VERL_PATH:-/work/wepsdl/projects/verl}
MCORE_PATH="${MCORE_PATH:-/root/Megatron-LM/}"
MBRIDGE_PATH="${MBRIDGE_PATH:-/work/wepsdl/mbridge}"
MEGATRON_BRIDGE_PATH="${MEGATRON_BRIDGE_PATH:-/root/Megatron-Bridge}"

export PYTHONPATH="$BOOTSTRAP_PATH:$VERL_PATH:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:${PYTHONPATH:-}"
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# deterministic compute (gcore apply_deterministic_mode parity). These env vars
# must be set BEFORE the process / NCCL communicators start. The torch-level flags
# and the FA3 unload happen in _bootstrap/sitecustomize.py (gated on
# GCORE_VERL_DETERMINISTIC); the per-field model/rollout switches live in the yaml.
export GCORE_VERL_DETERMINISTIC=${GCORE_VERL_DETERMINISTIC:-1}
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_DETERMINISTIC=1
export NCCL_ALGO=Ring
export FLASH_ATTENTION_DETERMINISTIC=1
export PYTHONHASHSEED=42

# gcore<->verl forward-input alignment via thd sequence packing (both sides use the
# same preprocess_packed_seqs impl). The bshd reshape hack is OFF by default (thd
# needs no padding/mask alignment); set VERL_ALIGN_GCORE=1 only for the bshd path
# (use_remove_padding=False). Qwen3-0.6B pad_token_id=151643.
# export VERL_ALIGN_GCORE=${VERL_ALIGN_GCORE:-0}
# export VERL_ALIGN_PAD_TO_MULTIPLE=${VERL_ALIGN_PAD_TO_MULTIPLE:-512}
# export VERL_ALIGN_PAD_TOKEN_ID=${VERL_ALIGN_PAD_TOKEN_ID:-151643}

# wandb: verl has no config field for host/key; it relies on env vars (see
# verl/utils/tracking.py -> wandb.init). Aligns gcore yaml report.wandb_host/key.
export WANDB_BASE_URL=
export WANDB_API_KEY=

# Multi-node Ray bootstrap. Only acts when LAUNCHER=mpirun; otherwise assumes a Ray
# cluster already exists (or single node). Run ONLY on the head machine. On success
# it sets the global NNODES (derived from the hostfile) and exports RAY_ADDRESS so
# main_ppo attaches to the cluster we just started.
#   usage: verl_ray_bootstrap "$NGPUS_PER_NODE"
verl_ray_bootstrap() {
    local ngpus="$1"
    LAUNCHER=${LAUNCHER:-local}
    HOSTFILE=${HOSTFILE:-/etc/mpi/hostfile}
    RAY_PORT=${RAY_PORT:-6379}
    TCP_IF=${TCP_IF:-bond1}
    SYNC_DEPS=${SYNC_DEPS:-True}
    PIP_DEPS=${PIP_DEPS:-"torchdata peft cachetools"}
    [ "$LAUNCHER" = mpirun ] || return 0

    # head ip on the cluster interface (this machine)
    HEAD_IP=${HEAD_IP:-$(ip -4 -o addr show "$TCP_IF" | awk '{print $4}' | cut -d/ -f1)}
    # one process per node; NNODES is derived from the hostfile (global, on purpose)
    local MPI_HOSTFILE=/tmp/ray_hostfile
    sed 's/slots=[0-9]*/slots=1/g' "$HOSTFILE" > "$MPI_HOSTFILE"
    NNODES=$(grep -cve '^\s*$' "$MPI_HOSTFILE")

    # sync python deps on every node; set SYNC_DEPS=False to skip once provisioned.
    if [ "$SYNC_DEPS" = True ]; then
        mpirun --allow-run-as-root \
            --bind-to none --map-by slot --hostfile "$MPI_HOSTFILE" \
            --mca btl_tcp_if_include "$TCP_IF" --mca oob_tcp_if_include "$TCP_IF" --mca routed direct \
            -x PATH \
            bash -c 'pip install --no-deps '"$PIP_DEPS"' >/tmp/sapo_dep_install.log 2>&1 || { echo "dep install failed on $(hostname)"; cat /tmp/sapo_dep_install.log; exit 1; }'
    fi

    mpirun -v --allow-run-as-root \
        --bind-to none --map-by slot --hostfile "$MPI_HOSTFILE" \
        --mca btl_tcp_if_include "$TCP_IF" --mca oob_tcp_if_include "$TCP_IF" --mca routed direct \
        -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x RAY_DEDUP_LOGS \
        -x VLLM_USE_V1 -x CUDA_DEVICE_MAX_CONNECTIONS \
        -x GCORE_VERL_DETERMINISTIC -x NVTE_ALLOW_NONDETERMINISTIC_ALGO -x CUBLAS_WORKSPACE_CONFIG \
        -x NCCL_DETERMINISTIC -x NCCL_ALGO -x FLASH_ATTENTION_DETERMINISTIC -x PYTHONHASHSEED \
        -x VERL_ALIGN_GCORE -x VERL_ALIGN_PAD_TO_MULTIPLE -x VERL_ALIGN_PAD_TOKEN_ID \
        -x WANDB_BASE_URL -x WANDB_API_KEY \
        bash -c '
            ray stop --force || true
            if [ "$OMPI_COMM_WORLD_RANK" = 0 ]; then
                ray start --head --node-ip-address '"$HEAD_IP"' --port='"$RAY_PORT"' --num-gpus '"$ngpus"'
            else
                sleep 5
                ray start --address='"$HEAD_IP":"$RAY_PORT"' --num-gpus '"$ngpus"'
            fi
        '
    # let main_ppo attach to the cluster we just started instead of a new local one
    export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"
}
