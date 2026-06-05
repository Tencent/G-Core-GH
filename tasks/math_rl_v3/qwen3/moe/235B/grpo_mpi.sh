PLACE_CFG_FOLDER=$PWD/place-config
LOG_DIR="$PWD/log/test_run"
PART=1

mkdir -p $LOG_DIR

head -n 16 /etc/mpi/hostfile >/root/hostfile
sed -i 's/slots=8/slots=1/g' /root/hostfile


#TODO: 既然这里设置了 tp 和 pp，那么直接把信息也放进 config 里，训练脚本直接读算了
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
      --sampler-tp-size 16 --sampler-pp-size 1 \
      --critic-tp-size 1 --critic-pp-size 4 \
      --actor-tp-size 2 --actor-pp-size 8 --actor-cp-size 4 --actor-ep-size 8 \


mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile $PLACE_CFG_FOLDER/sampler.hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
      bash tasks/math_rl_v3/qwen3/moe/235B/grpo_qwen3_235B.sh $PLACE_CFG_FOLDER sampler >$LOG_DIR/sampler$PART.log 2>&1 &

mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile $PLACE_CFG_FOLDER/critic.hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
      bash tasks/math_rl_v3/qwen3/moe/235B/grpo_qwen3_235B.sh $PLACE_CFG_FOLDER critic >$LOG_DIR/critic$PART.log 2>&1 &

mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile $PLACE_CFG_FOLDER/actor.hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
      bash tasks/math_rl_v3/qwen3/moe/235B/grpo_qwen3_235B.sh $PLACE_CFG_FOLDER actor >$LOG_DIR/actor$PART.log 2>&1 &
