PLACE_CFG_FOLDER=$PWD/place-config-qwen3-grpo-fp8
LOG_DIR="$PWD/log/qwen3_30b_fp8_all_gspo"
PART=1

mkdir -p $LOG_DIR

head -n 2 /etc/mpi/hostfile >/root/hostfile
sed -i 's/slots=8/slots=1/g' /root/hostfile

mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile /root/hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
      source ${PWD}/tasks/math_rl_v3/qwen/prepare_env.sh
sleep 3

#TODO: 既然这里设置了 tp 和 pp，那么直接把信息也放进 config 里，训练脚本直接读算了
mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile /root/hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
      pkill -9 -f python
sleep 3

RUN_SHELL=tasks/math_rl_v3/fp8_align/qwen3/gspo_30B_fp8_all.sh

mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile /root/hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
      python tools/auto_place.py \
      --fn gen --config-folder $PLACE_CFG_FOLDER \
      --sampler-tp-size 2 --sampler-pp-size 1 \
      --critic-tp-size 1 --critic-pp-size 1 \
      --actor-tp-size 4 --actor-pp-size 1 --actor-cp-size 2 --actor-ep-size 8 --actor-etp-size 1 \

mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile $PLACE_CFG_FOLDER/sampler.hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
      bash $RUN_SHELL $PLACE_CFG_FOLDER sampler >$LOG_DIR/sampler$PART.log 2>&1 &

mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile $PLACE_CFG_FOLDER/critic.hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
      bash $RUN_SHELL $PLACE_CFG_FOLDER critic >$LOG_DIR/critic$PART.log 2>&1 &

mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile $PLACE_CFG_FOLDER/actor.hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
      bash $RUN_SHELL $PLACE_CFG_FOLDER actor >$LOG_DIR/actor$PART.log 2>&1 &
