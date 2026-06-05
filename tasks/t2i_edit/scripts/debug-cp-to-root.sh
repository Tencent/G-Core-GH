cp /etc/mpi/hostfile /root/hostfile
sed -i 's/slots=8/slots=1/g' /root/hostfile

mpirun -v --allow-run-as-root \
  --bind-to none --map-by slot --hostfile /root/hostfile \
  --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
  rsync -av hf-hub/Qwen/Qwen-Image-Edit-2511 /root
