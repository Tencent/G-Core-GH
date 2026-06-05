# 一个不规范的临时测试脚本

train_ckpt=$1 # something like ../gcore-dev/t2i_grpo_tv4_oteam4_4_save_2/step-512
train_ckpt=`realpath $1`

ln -s ${PWD}/hf-hub/wechat/oteam4_4-step-10000/scheduler $train_ckpt/
ln -s ${PWD}/hf-hub/wechat/oteam4_4-step-10000/vae $train_ckpt/
mkdir -p $train_ckpt/transformer
ln -s ${PWD}/hf-hub/wechat/oteam4_4-step-10000/transformer_ema/config.json $train_ckpt/transformer/
ln -s $train_ckpt/diffusion_pytorch_model.bin $train_ckpt/transformer/
mkdir -p $train_ckpt/transformer_ema
ln -s ${PWD}/hf-hub/wechat/oteam4_4-step-10000/transformer_ema/config.json $train_ckpt/transformer_ema/
