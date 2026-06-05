PLACE_CFG_FOLDER=$PWD/place-config

cat /etc/mpi/hostfile > /root/hostfile
sed -i 's/slots=8/slots=1/g' /root/hostfile

# # 第二次注释可以取消下面 3 个 mpirun 命令
# mpirun -v --allow-run-as-root \
#       --bind-to none --map-by slot --hostfile /root/hostfile \
#       --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
#       -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
#       rm /usr/lib64/libstdc++.so.6

# mpirun -v --allow-run-as-root \
#       --bind-to none --map-by slot --hostfile /root/hostfile \
#       --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
#       -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
#       rm -rf /root/.cache/

# mpirun -v --allow-run-as-root \
#       --bind-to none --map-by slot --hostfile /root/hostfile \
#       --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
#       -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
#       ln -s /root/conda/lib/libstdc++.so.6.0.29 /usr/lib64/libstdc++.so.6

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
      --sampler-tp-size 4 \

mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile $PLACE_CFG_FOLDER/sampler.hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
      bash tools/eval/sh/eval_server.sh $PLACE_CFG_FOLDER > eval.log 2>&1 &
