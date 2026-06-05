#!/bin/bash
set -e

# 配置路径
PLACE_CFG_FOLDER=$PWD/place-config
LOG_DIR="${PWD}/../log/retool_log/retool_gcore_7b_external_agent_debug"
mkdir -p $LOG_DIR

echo $LOG_DIR
# 0. 环境准备 (Hostfile等)
cat /etc/mpi/hostfile >/root/hostfile
sed -i 's/slots=8/slots=1/g' /root/hostfile

# 2. 清理旧进程 (放在最前，避免把刚启动的 sandbox 也杀掉)
mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile /root/hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
      sh -c "pkill -9 -f '[p]ython' || true"
sleep 3

# 日志文件路径
export RETOOL_MONITOR_LOG_PATH="${LOG_DIR}/retool_quality_monitor.log"
export RETOOL_GEN_INTERVALS_DEBUG_LOG_PATH="${LOG_DIR}/gen_intervals_debug.log"
export RETOOL_MESSAGES_TRAJ_DEBUG_LOG_PATH="${LOG_DIR}/messages_traj_debug.log"
# 1. 启动本地沙箱 (Local Sandbox)
# 注意：这会在运行脚本的节点(通常是Master)启动沙箱。
# 如果Sampler分布在多机，需要确保Sampler能访问Master的IP，或者在每台机器都起沙箱。
# 这里假设是单机或Master可访问模式。
export SANDBOX_LOCAL_PORT=8008
# 获取本机IP (根据你的网络环境调整，如 bond1 或 eth0)
MY_IP=$(hostname -I | awk '{print $1}')
export SANDBOX_FUSION_URL="http://${MY_IP}:${SANDBOX_LOCAL_PORT}/run_code"

echo "Starting Local Sandbox at ${SANDBOX_FUSION_URL}..."
nohup python3 tasks/retool/local_sandbox_server.py > ${LOG_DIR}/sandbox.log 2>&1 &
echo "finish start sandbox"
SANDBOX_PID=$!

# 确保脚本退出时杀掉沙箱
trap 'kill $SANDBOX_PID' EXIT

# 等待沙箱启动
sleep 5

# 3. 安装依赖 (如果需要)
mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile /root/hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
      pip install jieba scikit-learn fastapi uvicorn
sleep 3

# 4. Auto Place (生成配置)
mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile /root/hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
      python tools/auto_place.py \
      --fn gen --config-folder $PLACE_CFG_FOLDER \
      --sampler-tp-size 2 --sampler-pp-size 1 \
      --critic-tp-size 1 --critic-pp-size 1 \
      --actor-tp-size 2 --actor-pp-size 1 --actor-cp-size 2

# 5. 启动各个角色
# 注意：你需要确保 SANDBOX_FUSION_URL 被传递给 Sampler
# mpirun 的 -x 选项可以传递环境变量

echo "Starting Sampler with SANDBOX_FUSION_URL=${SANDBOX_FUSION_URL}"

# 使用 ReTool 专用的 dapo_retool.sh 脚本
mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile $PLACE_CFG_FOLDER/sampler.hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x SANDBOX_FUSION_URL \
      bash tasks/retool/external_agent/dapo_retool.sh $PLACE_CFG_FOLDER sampler >$LOG_DIR/sampler$PART.log 2>&1 &

mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile $PLACE_CFG_FOLDER/critic.hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
      bash tasks/retool/external_agent/dapo_retool.sh $PLACE_CFG_FOLDER critic >$LOG_DIR/critic$PART.log 2>&1 &

mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile $PLACE_CFG_FOLDER/actor.hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
      bash tasks/retool/external_agent/dapo_retool.sh $PLACE_CFG_FOLDER actor >$LOG_DIR/actor$PART.log 2>&1 &

# 等待任务结束
wait
