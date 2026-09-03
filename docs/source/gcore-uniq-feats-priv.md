# G-Core Unique Features vs. Open-Source Community

本文档总结 G-Core 相较于开源社区（verl、slime、ROLL）的独特功能与技术优势。

G-Core 能快速响应微信内部的 LLM 训练相关的需求，包括模型支持、训练算法、性能优化等，一些合作案例如 AI 搜索中的 多 GenRM GRPO 训练，小程序代码生成的 agentic rl 训练， WeMM 模型的性能优化，视觉 T2I 模型的国产卡训练支持等，各团队如有需求欢迎联系 nrwu。


---

## 一、异步/同步训练 Trainer 优化

本模块聚焦训练流程与系统层面的优化，提升 GPU 利用率与训练吞吐。


### 1.1 Co-located Gen-RM: GPU 利用率翻倍

G-Core 实现了 **Co-located Generative Reward Model**（Gen-RM），将生成式 RM 与训练 actor 部署在同一组 GPU 上，通过 placement group 精细调度 GPU 分配，避免了传统方案中 RM 与 actor 分离部署导致的 GPU 空闲浪费。

- 在 Gen-RM 模式下，GPU 利用率**超出开源社区 100% 以上**。
- 支持多 RM 场景：`GenRmClient` 可管理多个 RM 实例（`gen_rm_config.reward_model_info`），每个 RM 独立分配 GPU 资源。
- 同时支持 LLM Gen-RM（`GenRmClient`）与 T2I Gen-RM（`T2iGenRmClient`）。
- 通过 `create_gen_rm_group` / `compute_gen_rm_placement` 实现灵活的 GPU 资源编排。


### 1.2 Async Rollout Training

支持**异步 rollout** 训练模式（`GrpoSingleCtrlTrainer`），rollout 和 training 可以 overlap 执行，进一步提升 GPU 利用率。

#### 动态资源摆放（开发中）

当前开源社区（verl、slime 等）普遍采用**静态资源切割**方案：训练前预先划分 rollout GPU 与 train GPU，运行期间固定不变。这种方式在 rollout 与 train 耗时不均衡时，会导致一侧 GPU 大量空闲等待。

G-Core 正在开发**动态资源摆放**能力：根据运行时 profiling 数据，在 rollout 与 train 阶段之间动态调整 GPU 分配比例，使资源利用率接近理论上限。

#### 树形采样（BETA）

VeRL rollout 采用 trajectory 采样：对每条 prompt 独立生成 N 条 response。这种方式在 advanttage 效率上存在瓶颈，无法控制条件概率。

G-Core 正在开发**树形采样**（Tree Sampling）策略：在 rollout 阶段以树状结构展开生成，每一步根据已有前缀分支探索不同方向，提升采样多样性与探索效率。参考 Gi-GPO、AT-GPO、TreePO 等前沿工作。

### 1.3 长尾优化：Partial Rollout、Rollpack、LASER

G-Core 针对 RL 训练中的长尾序列问题提供了多层次优化方案：

- **Partial Rollout**：对超长序列提前截断 rollout，避免少量长尾样本拖慢整个 batch 的生成速度（普遍共有）。
- **Rollpack (BETA)**：将多条短序列打包到同一 rollout slot 中，提升 GPU 利用率。尽管 ROLL 论文中提出了该概念，但实际代码并未开源。G-Core 是少数实际实现了 rollpack 的训练框架。
- **Profiling-based LASER**：基于 profiling 数据的长尾感知调度策略，为 async rollout 与多奖励（paper 投稿中）。


### 1.4 MoE Router Replay (R3)

为 MoE 模型训练实现了 **Router Replay** 机制：在 rollout 阶段记录 router 的 expert 选择决策，在 train 阶段 replay，保证 forward 的确定性，解决 MoE RL 训练中 rollout-train 不一致的问题。

G-Core 在 25 年 3 月 支持了 R3 feature，提前为 AI 搜提供了技术支持并落地，社区 26 年 3 月才支持了 feature。


### 1.5 故障自愈能力

大规模分布式训练中，硬件故障与训练卡死是常见问题。G-Core 提供了多层次的容错与自愈机制：

#### 训练卡死检测与自动重启

所有主要 Trainer 均继承 `TrainerRetryMixin`，支持训练卡死的自动检测与全量重启：

- **卡死检测**：每个 Actor 维护 `last_progress_time` 时间戳，每完成一个 train step 更新。`RayTrainGroup` 以 30 秒为周期轮询 actor 状态，若超过 `max_train_step_waiting_time` 未更新，则判定训练卡死。
- **自动重启**：检测到卡死后，执行 `orches.shutdown()` 关闭整个 Ray 编排层，等待完全关闭后重建训练 pipeline。最多重启 `max_restart_attempts` 次。
- **断点续训**：配合 `auto_load_from_save_ckpt` 配置，重启后自动从最近的 checkpoint 恢复训练，避免重复计算。

相关配置：`training.max_restart_attempts`、`training.max_train_step_waiting_time`

#### 硬件故障节点剔除（开发中）

通过 gemini 提供 GPU 健康检查工具能力，验证硬件可用性。当前为手动诊断工具，**自动检测 + 剔除 bad node + 动态扩缩容重启**能力正在开发中。


### 1.6 算子优化

#### Vocab-Parallel Entropy

自定义的 `_VocabParallelEntropy` autograd function，在 tensor parallel 下高效计算 per-token entropy（无需 all-gather 完整 logits），支持 entropy bonus 训练。开源社区的实现通常需要 all-gather，显存和通信开销更大。

#### Triton Fused Sparse Attention

实现了基于 Triton 的 fused sparse attention kernel，针对稀疏注意力模式进行融合优化，减少 kernel launch 开销与显存占用，提升长序列场景下的训练与推理效率。


### 1.6 双训练后端：Megatron-Core + FSDP2

G-Core 支持两种训练后端，可按需切换：

- **Megatron-Core Backend**：完整 3D 并行（TP/PP/DP/CP），适用于超大规模 LLM 训练。
- **PyTorch FSDP2 Backend**：原生 PyTorch 方案，适用于 T2I、DPO 等场景。

两种后端共享统一的 actor/trainer 接口（`base_engine.py`、`swap_mixin.py`），包括模型/优化器 offload/onload、checkpoint 管理等。


### 1.7 DP Balance（序列长度负载均衡）

实现了跨 DP rank 的序列长度 rebalance 机制（`dp_balance`），将不同长度的 sample 重新分配到各 DP rank，最小化 padding 浪费，提升训练效率。支持在 compute_log_probs 和 rl_train 阶段分别启用。


### 1.9 Dynamic Context Parallel（BETA）

当前 G-Core 已支持**静态 Context Parallel**（通过 `dist_config.context_parallel_size` 配置固定 CP 度），配合 Megatron-Core 的 CP 通信原语实现长序列的跨 GPU 分片。

在此基础上，G-Core 正在开发 **Dynamic Context Parallel** 能力：根据 batch 内序列长度的实际分布，动态调整 Context Parallel 的并行度。短序列无需拆分即可放入单 GPU，长序列则自动提升 CP 度以突破显存限制。相较于静态 CP 方案中"所有序列统一按最大 CP 度切分"的浪费，动态 CP 可显著减少不必要的通信开销，提升混合长度场景下的训练效率。


---

## 二、模型支持

本模块聚焦 G-Core 对不同模型架构和模态的支持能力。


### 2.1 mbridge: HF ↔ Megatron 桥接

G-Core 团队是 **mbridge**（HuggingFace ↔ Megatron checkpoint 转换）的主要贡献者。Slime 和 verl 中对许多模型的支持实际上依赖 mbridge 提供的转换能力。

G-Core 团队贡献的模型转换支持包括：

| 模型 | 时间 |
|------|------|
| GLM-4.5V | 2025-08 |
| Gemma 3 | 2025-09 |
| InternVL3 | 2025-09 |
| Qwen3-VL Dense | 2025-10 |
| Qwen3-VL-MoE | 2025-10 |
| InternVL3.5 | 2025-10 |
| WeLM v4 | 2026-01 |
| Qwen3.5 (Dense & MoE) | 2026-02 |
| WeLM v3 | 2026-03 |
| WeMM Audio | 2026-03 |

此外，团队还贡献了 tie_word_embeddings PP 支持、分布式文件系统 checkpoint 优化、权重保存优化等基础设施改进。

相关代码：`mbridge/`、`Megatron-Bridge/`


### 2.2 T2I (Text-to-Image) 模型支持

G-Core 原生支持 **Text-to-Image 模型的 RL 训练**（DiT GRPO / DPO），这在开源社区极为罕见：

- `T2iGrpoTrainer`：T2I 模型的 GRPO 训练
- `T2iDpoTrainer` / `BaseDpoTrainer`：T2I 模型的 DPO 训练
- `T2iEditSftTrainer`：T2I 编辑模型的 SFT 训练
- T2I 专用的 Gen-RM Client (`T2iGenRmClient`)

相关代码：`gpatch_v4/trainer/t2i_grpo_trainer.py`、`gpatch_v4/trainer/t2i_dpo_trainer.py`


---

## 三、Trainer 支持

本模块聚焦 G-Core 提供的丰富训练范式与 Agentic RL 基础设施。


### 3.1 丰富的 Trainer 矩阵

G-Core 提供了覆盖多种训练范式的 trainer：

| Trainer | 说明 |
|---------|------|
| `GrpoTrainer` | LLM GRPO 训练 |
| `GrpoSingleCtrlTrainer` | 单控制器 GRPO（colocate / disaggregated） |
| `DpoTrainer` | LLM DPO 训练 |
| `FinetuneTrainer` | SFT / 微调 |
| `OnPolicyDistillTrainer` | 在线蒸馏 (OPD / G-OPD) |
| `OffPolicyDistillTrainer` | 离线蒸馏 |
| `T2iGrpoTrainer` | T2I GRPO 训练 |
| `T2iDpoTrainer` | T2I DPO 训练 |
| `T2iEditSftTrainer` | T2I 编辑 SFT |
| `BagelTrainer` | Bagel 多模态训练 |
| `OmniBaseTrainer` | 通用多模态基础 trainer |


### 3.2 Agentic RL 与 Agent 树形采样

G-Core 拥有完整的 **Agentic RL** 基础设施，支持：

- **TrajEnvManager**：管理多轮交互式 trajectory 环境，支持 tool calling、代码沙箱执行等 agent 场景。
- **多种 Env 类型**：
  - `RetoolDapoEnv`：Math ReTool + 代码沙箱交互环境
  - `InteractiveCliEnv`：CLI 交互式环境
  - `MiniProgram Env`：小程序交互环境
  - `Sokoban Env`：推箱子游戏环境（用于 agent 能力评测）
- **Tool Parser**：支持 sglang 等引擎的 tool call 解析。
- **LLM Proxy 层**：抽象推理引擎接口（sglang proxy / engine proxy），支持 agent 多轮生成。
- **树形采样 (BETA)**：支持 agent 场景下的多步 tree search 采样策略。
- **GrpoAgenticTrainActor**：专用的 agentic 训练 actor，支持 step-level reward、GAE advantage 估计、trajectory 级别的数据处理。

相关代码：`gpatch_v4/agentic/`、`gpatch_v4/actor/grpo_agentic_train_actor.py`、`tasks/retool/agentic_rl/`


---

## 四、LM / VLM 算法支持

聚焦 G-Core 在 RL 算法层面的支持。开源社区对各类 RL 算法的实现散落于不同仓库，质量参差不齐且缺乏统一抽象。G-Core 通过 `LOSS_FUNC_REGISTRY` 提供了可插拔的 loss 函数注册机制，将主流与前沿算法整合至统一框架下，降低算法切换与组合的成本。


### 4.1 GSPO (Group Sparse Policy Optimization)

- 参考：[arXiv:2507.18071](https://arxiv.org/pdf/2507.18071)
- 实现 token-level 与 sequence-level 的 GSPO ratio 计算。
- 注册名称：`@register_loss("gspo")`


### 4.2 FIPO (Future-KL Influenced Policy Optimization)

- 参考：[arXiv:2603.19835](https://arxiv.org/abs/2603.19835)
- 基于 Future-KL 的影响力权重，通过指数衰减因子 gamma = 2^(-1/tau) 计算 token 间的影响传播。
- 支持 chunked 矩阵乘法（`fipo_chunk_size`）以控制显存。
- 支持 asymmetric clip、safety threshold、correction-aware filter 等高级配置。
- 注册名称：`@register_loss("fipo")`


### 4.3 G-OPD (Generalized On-Policy Distillation)

- 参考：[arXiv:2602.12125](https://arxiv.org/abs/2602.12125)
- 在标准 OPD 基础上引入 `g_opd_lambda` 参数，支持 G-OPD (lambda < 1.0)、标准 OPD (lambda = 1.0)、ExOPD (lambda > 1.0, reward extrapolation) 三种模式。
- 支持 base model 作为 reference model 进行 reward correction。
- 支持多 teacher routing（`g_opd_teacher_routing_field`）。
- 支持 mix reward advantage 与 reverse KL advantage。
- 注册名称：`@register_loss("opd")`


### 4.4 GDPO (Group DPO with Multi-Reward)

- 支持多维 reward 的加权 advantage 计算（`gdpo_reward_weights`）。
- 每个 reward 维度独立计算 GRPO advantage，按权重合并。
- 函数：`calculate_gdpo_advantages()`


### 4.5 DAPO 系列增强

- 实现 **Clip-Higher**（asymmetric clip ratio）：`ppo_clip_ratio_low` / `ppo_clip_ratio_high`
- 实现 **Overlong Penalty**：对超长生成施加惩罚（`dapo_overlong_penalty`）
- 参考：[verl DAPO recipe](https://github.com/volcengine/verl/blob/main/recipe/dapo/README.md)


### 4.6 Dual-Clip PPO

支持 Dual-clip PPO（[arXiv:1912.09729](https://arxiv.org/pdf/1912.09729)），通过 `ppo_dual_clip_ratio_c` 参数在标准 PPO clip 基础上增加第二重 clip 约束。开源框架少有完整支持。


### 4.7 Off-Policy Correction（重要性采样框架）

G-Core 实现了完整的 off-policy correction 框架，支持多种模式：

- **聚合层级**：`token` / `sequence` / `geometric` 三种 importance weight 粒度
- **处理模式**：
  - `truncate` (TIS)：截断到上界
  - `mask` (MIS)：超出范围则 zero + 修改 mask 分母
  - `icepop`：兼容 verl IcePop 实现
  - `clip` (CIS)：clip 到 [lower, upper]
- **Per-token veto**：任一 token ratio 低于阈值则整条 sequence 梯度归零

相关代码：`gpatch_v4/core/correction_helper.py`、`gpatch_v4/configs/ppo_config.py`
