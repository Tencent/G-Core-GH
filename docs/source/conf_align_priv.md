# 配置对齐

这个文档会讲述 verl 与 gcore 几个常见的配置 diff 与迁移方式。
这是一个内部的文档（直接对比可能违反规制）。

## 基本概念

### 并行

Megatron 包含了非常复杂的并行策略，算法一般只需要知道 DP-size（data parallel size）数据并行度。

例如你有 $N_\rm{gpu}=1024$ 卡，如果你是普通的 transformers 训练，那么你的 dp-size 就是 1024。

而如果你用了 megatron，你选择了 context parallel size 4，tensor parallel size 2，那么你的 dp size
就是 $S_\rm{dp} = \frac {1024} {2 \times 4} = 128$。

### MBS（micro batch size）和 GBS（global batch size）

MBS 是一次 forward / backward，或者说，一个 MB（micro batch），的数据量大小。
受到显存限制，通常来说比较小，比如 1，最多2。
极端情况下，非常小的模型（如 0.5B），非常短的 seqlen（<1k），可能可以达到 32 之类的夸张数字。

GB（global batch）包含多个 MB $S_\rm{mb}$，他的大小是 GBS（global batch size）$S_\rm{gb}$。
一次 forward / backward 不足以消费 GBS 数量的数据，就需要引入 GAS（Gradient Accumulation Step）：

$$
S_\rm{ga} = \frac {S_\rm{gb}} {S_\rm{dp} \times S_\rm{mb}}
$$

由于 micro batch 不发生 grad all-reduce，也不发生 optimizer
step，所以他的数字理论上完全不影响模型的收敛。但由于 tensorcore 的硬件因素，加上 op implementation
的归约因素，会有轻微的 diff，一般不影响。

**算法真正关心的是 GBS，GBS 才会影响模型的收敛。MBS、GAS 这些理论上都不影响**，而工程上的
diff，只要实现正确，几乎没有影响。

### Rollout GBS 与 Train GBS

强化学习与普通训练最大的区别在于 rollout 过程，也就是说所谓的 on policy。RL
中不需要答案标注，只需要问题，在 rollout 阶段，根据 prompt，采样（sampling）出来多个
response。再用这些 response 和 reward 来训练。

因此，这里引入一个新的 batch size，就是 rollout 的 global batch size $S_\rm{rgb}$。

并不是说一次 rollout 就是一次 train，实际上，PPO 论文里一个点就是用一次 rollout
的结果，进行多次训练。

例如 $S_\rm{rgb}=256$，repeat 是 $8$，那么实际上会产生 $256 \times 8 = 2048$ 条数据。如果你的 train
GBS $S_\rm{gb}=128$，那么实际上会有 $\frac {2048} {128} = 16$ 个 global batch，也就是 $16$ 次
optimizer step 更新。

## verl 对齐

verl 有些地方的命名比较特殊。

### Step

verl 的 step 默认是指 PPO step（rollout），而 gcore 的 step 默认是指 train iter。

### GBS

verl 的 rollout GBS 名字是 `data.train_batch_size`，对应 gcore `--rollout-global-batch-size`。

verl 无法直接设置 train GBS，而是 train GBS = `actor_rollout_ref.actor.ppo_mini_batch_size` $\times$ `actor_rollout_ref.rollout.n`。

你可以根据这个关系反推 gcore 的参数设置。

### Epoch

`--ppo-max-epochs` 是指数据会过多少轮，和 sft 概念相同。在整个 dataloader 的维度来看的，理解为数据总共要过多少个 epoch，全部数据完整过完之后再重新来一遍；

`--ppo-max-epochs-2` 是指一次 rollout（一个 ppo step），数据会用来训练多少次 train。指一批 rollout global batch 的数据要重复过多少轮。这是在 rollout global batch 维度来看，当前 rollout global batch 过完之后，再把同一批 rollout global batch 过一遍。在走完 epochs2 轮之后再 rollout 下一批数据接着训练。

### 数据过滤

verl 默认会过滤数据，gcore 默认不过滤数据，gcore 过滤数据由用户自定义规则。
