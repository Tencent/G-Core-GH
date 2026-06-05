PLACE_CFG_FOLDER=$PWD/place-config-gen-rm
LOG_DIR="$PWD/log/qwen3vl-gen-rm"
PART=1

USE_GSPO=1

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
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
      python tools/auto_place.py \
      --fn gen --config-folder $PLACE_CFG_FOLDER \
      --sampler-tp-size 4 --sampler-pp-size 1 \
      --gen-rm-tp-size 16 --gen-rm-pp-size 1 \
      --actor-tp-size 2 --actor-pp-size 2 --actor-cp-size 1 --actor-ep-size 16 \
      --ray-port 6789

mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile $PLACE_CFG_FOLDER/sampler.hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
      bash tasks/qwen3vl/gen-rm/qwen3vl_moe_grpo_gen_rm.sh $PLACE_CFG_FOLDER sampler $USE_GSPO >$LOG_DIR/sampler$PART.log 2>&1 &

mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile $PLACE_CFG_FOLDER/gen-rm.hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
      bash tasks/qwen3vl/gen-rm/qwen3vl_moe_grpo_gen_rm.sh $PLACE_CFG_FOLDER gen-rm $USE_GSPO >$LOG_DIR/gen-rm$PART.log 2>&1 &

mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile $PLACE_CFG_FOLDER/actor.hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
      bash tasks/qwen3vl/gen-rm/qwen3vl_moe_grpo_gen_rm.sh $PLACE_CFG_FOLDER actor $USE_GSPO >$LOG_DIR/actor$PART.log 2>&1 &
