
LOG_DIR="$PWD/log/glm4p5vl-sft"
PART=1

mkdir -p $LOG_DIR

cat /etc/mpi/hostfile > /root/hostfile
sed -i 's/slots=8/slots=1/g' /root/hostfile

export _MASTER_ADDR=$__POD_IP__

mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile /root/hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
      pkill -9 -f python
sleep 3


mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile /root/hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x _MASTER_ADDR  \
      bash tasks/glm4v/sh/train_glm4vl_sft.sh  >$LOG_DIR/sft$PART.log 2>&1 &
