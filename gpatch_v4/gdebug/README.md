# gdebug — RL 数据路径的正确性检查

RL 的数据路径上有一类错误不会让训练中断，梯度和指标照常产出，只是数值已经算错了。这个
目录放的是针对这类问题的提前检查：检查不修改训练状态，默认 `off`，由 `debug` 配置打开。

目前只有一条：`ppo_padding_check.py`，查 PPO 的 policy 与 critic 之间有没有 padding 长度
错位的风险。

## 启用

```yaml
debug:
  ppo_padding_check_mode: warn   # off | warn | abort
```

默认 `off`。`warn` 只在 rank 0 记一条告警并继续跑，`abort` 在 critic engine 构造之前抛
`RuntimeError`。其他取值会被拒绝：配置构造时 `assert`，`python -O` 下改由检查在运行时
`raise ValueError`。

## 检查的内容

检查从 `GrpoTrainActor.init` 调用，位置在 critic engine 构造之前，仅在
`advantage_type='ppo'` 且训练后端是 `mcore` 时执行。它针对的是 MCore 的
`_align_values_to_rollout_logprobs` 这条路，FSDP2 不在覆盖范围内。

能仅凭配置判定为非法的组合，由 `RlConfig.__post_init__` 直接拒绝；这里查的是仅凭配置无法
确认、但存在 padding 错位风险的组合，是否真的错位还取决于运行时的样本长度。

以下两种配置会命中检查：

1. **critic 开了 smart padding 而 policy 没开。** critic 这边按 forward-only microbatch
   分组内的最长序列补齐，policy 那边补到全局最大长度，所以 critic 的 value 可能比与它对齐的
   policy log-prob 短。反过来（policy 开、critic 不开）value 只会更长，当前的对齐逻辑用右
   截断就能处理，所以不命中。
2. **两侧都开 smart padding，但 `forward_only_mbs` 不同。** smart padding 取的是分组内的
   最长长度；分组不同之后，同一条样本可能在两侧被补到不同长度。

**命中只表示存在风险，是否真的错位仍取决于样本长度。**
`McoreEngine._align_values_to_rollout_logprobs` 会拿 prompt 和 response 的长度试着把 value
补回到 log-prob 的长度：有些样本本来就等长；有些两条补零规则都用不上，抛 `RuntimeError`；
还有些补回了长度并继续往下走，而这时 token 的坐标已经错了：命中的成因是两侧补齐宽度不同，
value 本身并没有少一段前缀，所以补进去的零会把每个 value 整体往后错开同样多的位置。那条
对齐函数只看长度，识别不出这种语义错位。仅凭静态配置分不出一次 run 会落到哪一种。

开了 `ppo.skip_prev_logps` 或 `policy.dist_config.dynamic_context_parallel` 时，policy
log-prob 的长度不由 policy engine 的 padding 决定，本检查直接跳过。

## 当前未覆盖

`policy.balance_dp_seqlen=True`：policy 的 log-prob 会在 DP 组内按长度重新分配、算完再还原，
而 critic 的 value 仍按原来的分配算，所以即使两侧的 padding 配置相同也可能对不上，本检查发现
不了这类情况。

## 分布式安全约束

- 除了读本检查的 mode 并校验它，`off` 时不读配置里的其他字段，也不做张量运算、host-device
  同步或集合通信。
- 挂点的可达性要和检查里的假设对上。本检查挂在 `require_critic_model()` 里面，那是
  `config.ppo.advantage_type in ["ppo"]`（`actor/mixin.py:1078-1085`）——读一个不可变的配置
  字段，没有任何 rank 本地的输入。整个 job 的每个 rank 都是从同一份 `RlConfig` 构造的，
  所以这个条件处处相同，整组要么都跑、要么都不跑。这条依赖各 rank 配置一致这个前提，
  和 `__post_init__` 那一整套校验依赖的是同一个。条件一旦读到 rank 本地的东西（张量、本地
  batch、设备状态），这个前提就不成立了，下面两条也跟着失效。
- 先分清结论是全组一致的还是 rank 本地的。rank 本地的问题不能只在 rank 0 记录，因为命中的
  未必是 rank 0；写「rank 0」的时候要说清是哪个 group 的 rank 0。
- `abort` 只能由全组一致的结论触发。rank 本地的条件要先让所有 rank 无条件参与同一次汇总、
  再按汇总结果一起退出，否则可能一个 rank 先抛异常、其余 rank 卡在后续的集合通信上。真用到
  集合通信时，所有参与 rank 要以相同的顺序、相同的次数、相同的 shape/dtype/device 进入。
- 这个目录下的模块在 import 时不初始化分布式状态、不探测设备、不引入只有某个后端才装的
  可选依赖——它会被 actor 无条件 import，那时这些东西还不一定就绪。

## 再加一条检查

- 新建一个模块，写一个纯判断函数（只读配置或状态、返回诊断，不改任何东西）加一个
  `maybe_check_*` 包装，在使用点之前调用。**不要**为此建 registry、基类或自动发现：各条检查
  的挂点时机、入参和 process group 本来就不一样，抽象只会把这些差异藏进间接调用里。
- 开关加成 `DebugConfig` 上的 `*_check_mode` 字段，取值校验写进**已有的**
  `__post_init__`。再定义一个同名的 `__post_init__` 会静默覆盖掉前一个，把上一条检查的校验
  一起带走——`test_gdebug_ppo_padding_check.py` 里有一条用例专门盯着这件事。
