# 训练资源预估

## 1. 显存
假设模型大小是 $\Phi$，actor 训练时每张 GPU 占用的显存约为

$$\Phi \cdot ( \frac{\mathrm{weights} + \mathrm{grads} + \mathrm{buffers}}{\mathrm{mp\_size}} + \frac{\mathrm{optimizer\_states}}{\mathrm{num\_gpus}} )
=\Phi \cdot (\frac{2 + 6 + O}{\mathrm{mp\_size}} + \frac{12}{\mathrm{num\_gpus}})$$

其中，$\mathrm{weights}$ 是 BF16，$\mathrm{grads}$ 是 BF16 + FP32，$\mathrm{optimizer\_states}$ 假设是 Adam 类优化器（FP32 的 w + m + v），$\mathrm{mp\_size}$ 是 actor 模型并行（TP + PP）的 GPU 数量，$\mathrm{num\_gpus}$ 是总 GPU 数量, $O$ 是 DDP buffers 等额外开销，约为 [0.5, 2]。

## 2. 吞吐
> 参考：https://arxiv.org/pdf/2302.13971
When training a 65B-parameter model, our code processes around 380 tokens/sec/GPU on 2048 A100 GPU with 80GB of RAM. This means that training over our dataset containing 1.4T tokens takes approximately 21 days.

当 model=65B, GPU=A100, seq_len=2048 时，吞吐 base_throughput = 380 tokens/sec/GPU

多个因素会影响吞吐，假设初始 throughput_coef = 1.0

- 考虑不同 GPU：H20 throughput_coef *= [0.5, 0.75], H100 throughput_coef *= [2, 2.3]
- 考虑模型大小 $\Phi$ ：throughput_coef *= $\frac{65B}{\Phi}$
- 考虑序列长度 $s$ ：throughput_coef *= $(\frac{2048}{s})^2$
- 考虑 MoE：e.g., 30B A3B 可以当 3B 的 dense 计算
- 考虑开 EP：跨机 throughput_coef *= 0.5， 单机 throughput_coef *= 0.8
- 考虑开 CP：throughput_coef *= cp\_decay, where cp\_decay $\in$ [0.6, 0.9]

最终吞吐大概是 throughput = base_throughput * throughput_coef

## 3. 并行策略选择

一些启发式规则：
- 先尽量把 MP (TP + PP) 开大，确定当前资源量能够跑起训练
- 然后把 MP 减小，尽量开大 DP 加速训练
- TP 通信量较大，tp_size 最好不要超过单机 GPU 数量，限制在节点内通信
- PP 开太大容易有 bubble
- 7B model 一般用 TP=1，13B model 用 TP=2 即可训练
- Megatron 主推方案是 FSDP + EP + CP （目前不成熟，建议等成熟后再部署测试）