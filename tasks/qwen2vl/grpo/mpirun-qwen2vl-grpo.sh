PLACE_CFG_FOLDER=$PWD/place-config
LOG_DIR="$PWD/log/qwen2vl"
PART=1

mkdir -p $LOG_DIR

cat /etc/mpi/hostfile > /root/hostfile
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
      --critic-tp-size 1 --critic-pp-size 4 \
      --actor-tp-size 2 --actor-pp-size 2 --actor-cp-size 2 \

mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile $PLACE_CFG_FOLDER/sampler.hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
      bash tasks/qwen2vl/grpo/qwen2vl_grpo.sh $PLACE_CFG_FOLDER sampler >$LOG_DIR/sampler$PART.log 2>&1 &

mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile $PLACE_CFG_FOLDER/critic.hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
      bash tasks/qwen2vl/grpo/qwen2vl_grpo.sh $PLACE_CFG_FOLDER critic >$LOG_DIR/critic$PART.log 2>&1 &

mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile $PLACE_CFG_FOLDER/actor.hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
      bash tasks/qwen2vl/grpo/qwen2vl_grpo.sh $PLACE_CFG_FOLDER actor >$LOG_DIR/actor$PART.log 2>&1 &
