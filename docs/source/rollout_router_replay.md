rollout router replay (R3)
======================

rollout router replay 能改善 moe 模型 rl 训练收敛稳定性， 这是目前业界的一个共识。 
我们率先在 gcore 支持了该项功能，并验证了该feature 的收益。 
我们实验发现 R3 确实可以降低训推引擎输出结果概率分布之间的 kl 散度，降低幅度大约为一半。


## 配置
只需要开启一个 flag

```
--moe-router-replay
```

## 镜像
目前 R3 支持 (推理返回 router index) 还未成为 sglang 正式发布的 feature, 我们使用了一个临时版本，打了一个临时镜像。预计 sglang 0.5.7 会正式支持 R3.

mirrors.tencent.com/wepsdl/rl-sglang-r3:v2.20.13.8-cuda-12.9-cudnn-9-py-3.10-torch-2.8.0-fa-2.8.1-te-2.8-sglang-0.5.4.post3

##  rollout 数据链路适配
默认的 model provider 以及 gen_rollouts  是支持 R3 的。 但是有很多业务可能定制化了 rollout (sampler 与 actor 之间) 数据链路，需要额外适配。请与gcore对接人员确认，是否需要适配。