model_dir=$1

cd $model_dir

for i in `ls *.safetensors`; do
  echo $__HOST_IP__
  dd if=$i of=/dev/zero bs=1M count=1024 oflag=dsync
done
