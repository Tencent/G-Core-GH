PLACE_CFG_FOLDER=$PWD/place-config
LOG_DIR="$PWD/log/qwen_tool_calling"
PART=1

mkdir -p $LOG_DIR

cp /etc/mpi/hostfile /root/hostfile
sed -i 's/slots=8/slots=1/g' /root/hostfile

mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile /root/hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
      pkill -9 -f python
sleep 3

mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile /root/hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
      python tools/auto_place.py \
      --fn gen --config-folder $PLACE_CFG_FOLDER \
      --sampler-tp-size 2 --sampler-pp-size 1 \
      --critic-tp-size 2 --critic-pp-size 2 \
      --actor-tp-size 2 --actor-pp-size 2 --actor-cp-size 2 \

mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile $PLACE_CFG_FOLDER/sampler.hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
      bash tasks/math_rl_v3/qwen-tool-calling/grpo.sh $PLACE_CFG_FOLDER sampler >$LOG_DIR/sampler$PART.log 2>&1 &
SAMPLER_PID=$!

mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile $PLACE_CFG_FOLDER/critic.hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
      bash tasks/math_rl_v3/qwen-tool-calling/grpo.sh $PLACE_CFG_FOLDER critic >$LOG_DIR/critic$PART.log 2>&1 &
RM_PID=$!

mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile $PLACE_CFG_FOLDER/actor.hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
      bash tasks/math_rl_v3/qwen-tool-calling/grpo.sh $PLACE_CFG_FOLDER actor >$LOG_DIR/actor$PART.log 2>&1 &
ACTOR_PID=$!

# echo 'wait for workers...'
# wait $SAMPLER_PID $RM_PID $ACTOR_PID
