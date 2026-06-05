#!/usr/bin/env bash
# Launch ReTool DAPO agentic training (TrajEnvManager + tasks.retool.async_agent_loop RetoolDapoEnv).
# Uses Hydra config search path: tasks/async_agent_loop (rl_config) + tasks/retool/async_agent_loop (this stack).
# set -euo pipefail
RCDIR="/work/wepsdl"
export PYTHONPATH="$RCDIR/gcore/gcore_agentic_rl:$RCDIR/gcore/gcore_agentic_rl/tests:$RCDIR/gpatch_v4:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"

export WANDB_BASE_URL=
export WANDB_API_KEY=
readonly LOG_DIR=/mnt/ceph-hz1-csp/mm-base-plt2/user_yeazhao/log/async_agent_loop/retool_dapo_test
# 检查log目录是否存在，不存在则创建
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi
source tests/test_gpatch_v4/mpirun-stop-ray.sh
source tests/test_gpatch_v4/mpirun-init-ray.sh
# 1. 启动本地沙箱 (Local Sandbox)
# 注意：这会在运行脚本的节点(通常是Master)启动沙箱。
# 如果Sampler分布在多机，需要确保Sampler能访问Master的IP，或者在每台机器都起沙箱。
# 这里假设是单机或Master可访问模式。
export SANDBOX_LOCAL_PORT=8008
# 获取本机IP (根据你的网络环境调整，如 bond1 或 eth0)
MY_IP=$(hostname -I | awk '{print $1}')
# 使用可路由 IP：Ray Train Actor 进程内默认拿不到 shell 的 SANDBOX_*，由 train_group 注入；多机时各节点需能访问该地址。
export SANDBOX_FUSION_URL="http://${MY_IP}:${SANDBOX_LOCAL_PORT}/run_code"

echo "Starting Local Sandbox at ${SANDBOX_FUSION_URL}..."
nohup python3 tasks/retool/local_sandbox_server.py > ${LOG_DIR}/sandbox.log 2>&1 &
echo "finish start sandbox"
SANDBOX_PID=$!

# 等待沙箱端口可连（避免固定 sleep 后仍 refused）
SANDBOX_READY=0
for _i in $(seq 1 30); do
  if python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', int('${SANDBOX_LOCAL_PORT}'))); s.close()" 2>/dev/null; then
    SANDBOX_READY=1
    echo "Sandbox listening on port ${SANDBOX_LOCAL_PORT}"
    break
  fi
  sleep 1
done
if [ "$SANDBOX_READY" != "1" ]; then
  echo "ERROR: sandbox did not open port ${SANDBOX_LOCAL_PORT}; see ${LOG_DIR}/sandbox.log"
  exit 1
fi

# # 确保脚本退出时杀掉沙箱
# trap 'kill $SANDBOX_PID' EXIT
python3 tasks/retool/async_agent_loop/retool_async_agent_loop.py \
  --config-path="." --config-name="retool_dapo_agentic_colocate.yaml" >$LOG_DIR/all_rank.log 2>&1 &
