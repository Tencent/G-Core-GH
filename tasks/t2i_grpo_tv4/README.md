# README

一个 t2i 的 demo。

数据：https://github.com/XueZeyue/DanceGRPO/blob/main/assets/prompts.txt 的 head。

# 数据

简单的 txt 或者 json，demo 无需预处理。已经下载在 demo 中。

无需离线处理 text encoder，on the fly 自动处理。

# 训练

下载需要的 checkpoint 和数据

```
bash tests/download.sh
```

修改配置修改 `tasks/t2i_grpo_tv4/yaml/flux_rl_config.yaml` 即可。

运行
```
bash tasks/t2i_grpo_tv4/scripts/grpo_flux.sh
```

# 训练闭源 oteam4-4

先把 checkpoint 放到 `$PWD/hf-hub/wechat` 下面，例如

```
tree hf-hub/wechat/oteam4_4-step-10000
hf-hub/wechat/oteam4_4-step-10000
│   └── scheduler_config.json
├── transformer
│   ├── config.json
│   └── diffusion_pytorch_model.bin
```


下载需要的 checkpoint 和数据

```
bash tests/download.sh
```

HPSv3 的代码写得不够健壮，所以只能做一个特定的环境：
```
pip3 install hpsv3==1.0.0
pip3 install transformers==4.51.3 peft==0.10.0
```

修改配置修改 `tasks/t2i_grpo_tv4/yaml/oteam4_4_rl_config.yaml` 即可。

运行
```
bash tasks/t2i_grpo_tv4/scripts/grpo_oteam4_4.sh
```

# 一些配置的解释

danceGRPO 的逻辑是 deepspeed 的习惯逻辑：
train micro batch size 用户给，
train gradient accumulation step 用户给，
然后 global batch size = train MBS * train GAS * dp SIZE，
这样有个问题是，如果用户加卡 ，GBS 会变，算法的逻辑会被破坏。

我们代码的逻辑是 megatron 的习惯逻辑：
train micro batch size 用户给，
train global batch size 用户给（影响算法效果），
然后 train GAS = train GBS / (GAS * dp size)，自动调整，
如果用户加卡 ，GBS 不变，算法的逻辑不会被破坏。

关键是：micro batch size 和 gradient accumulation step
都是工程参数，只会影响速度和显存，不影响算法效果，只有 global batch size
会影响，所以只要关心这里就好。

例子：

danceGRPO
```
mbs = 1 gas = 4 dp size = 8 -> gbs = 1 * 4 * 8 = 32
```

gcore
```
mbs = 1 gas = 32 dp size = 8 -> gas = 4
```
