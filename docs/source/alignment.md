# 确定性训练与 verl 对齐

**最后更新**：2026-07

## 概述

默认情况下，gcore 的 RL 训练**不是**逐比特可复现的：GPU kernel、请求调度、
batch 组成等环节的非确定性，会让同一份配置跑两遍得到不同的 reward 曲线。

- **Reproducibility**: 开启确定性模式后，gcore 两次相同配置的 run 可以达到产出**逐比特一致**的曲线
- **Alignment**同时，配合本页的对齐配置，gcore 与开源框架 [verl](https://github.com/volcengine/verl) 做对齐。
**NOTE: 跨框架长期 bitwise 对齐不现实**：训练侧 bf16 数值噪声会随 step 级联放大，跨框架长期逐比特一致不可期。
对齐目标是**曲线趋势一致**，而非无限步 bitwise。

适用场景：

- **调试复现**：逐 step 精确重现一次训练异常。
- **回归测试**：验证一处改动对训练结果没有静默影响。
- **跨框架对拍**：以 verl 为参照系，验证 gcore 的算法/后端实现正确。

**当前覆盖范围**：SGLang（rollout）+ Megatron/mcore（训练）+ GRPO + Qwen3-0.6B
dense + THD 序列 packing + TP2PP2CP2，其它后端 / 范式 / 模型见文末 [Roadmap](#roadmap)。

## 快速开始

### 0. 准备 verl 对照环境（仅对拍时需要）

> verl 需要 patch 原因参考 **How it works -> 对齐对 verl 的改动** 章节

verl 是独立的开源仓库。为对齐所做的唯一必要改动（`agent_loop.py` 的 GRPO 组内
`rollout_n` / 逐请求 seed 修复）以 patch 形式维护在
[patches/agent_loop.patch](../../tests/test_alignment_v4/common/verl/patches/agent_loop.patch)。
用现成脚本一次性完成 clone + 切到对齐所基于的 commit + 打 patch（幂等，重复运行安全）：

```bash
# 从仓库根目录执行；默认 clone 到 /work/wepsdl/projects/verl
bash tests/test_alignment_v4/rl/verl/scripts/setup_verl.sh

# 若想放到别处，用 VERL_PATH 指定（运行 verl 时也要用同一个 VERL_PATH）
VERL_PATH=/your/path bash tests/test_alignment_v4/rl/verl/scripts/setup_verl.sh
export VERL_PATH=/your/path
```

脚本默认的 clone 路径与 `common/verl/env.sh` 的 `VERL_PATH` 默认值一致
（`/work/wepsdl/projects/verl`），因此默认情况下运行时能直接对应上；改了路径就把
`VERL_PATH` 一并导出。env.sh 本身不再自动打 patch，只假设 `VERL_PATH` 已是打好补丁的
verl。只跑 gcore（自身复现）时可跳过本步骤。

### 1. 开启确定性 + 对齐（gcore 配置）

在 RL 配置里打开下列开关（对应 `RlConfig`，可写进 yaml 或用 hydra 覆盖）：

```yaml
training:
  apply_deterministic_mode: True   # torch 确定性 + 训练侧确定性 kernel
  attention_backend: flash         # 训练侧 attention 后端（TE flash）
  auto_load_from_save_ckpt: False  # 不自动 resume 旧 ckpt，保证从 HF base 起跑

data:
  shuffle: False                   # 固定样本顺序（与 verl 对齐 / 可复现）

policy:
  ppo_pack_seq: True               # THD 序列 packing（与 verl use_remove_padding 对齐）
  override_transformer_config:
    calculate_per_token_loss: True # 梯度按全局 token 数归一（token-mean，对齐 verl）

ppo:
  advantage_type: grpo
  ppo_ratio_eps: 0.2
  ppo_dual_clip_ratio_c: 20
  grpo_kl_loss_beta: 0
  ppo_entropy_bonus: 0
```

SGLang rollout 侧显式对齐（`sampler.infer_engine_configs[i]`）：

```yaml
sampler:
  infer_engine_configs:
    - attention_backend: flashinfer   # 推理侧后端
      disable_cuda_graph: False
      max_running_requests: 128        # 与 verl 显式对齐（verl 默认 1024）
      temperature: 1.0
      top_p: 1.0
```

其余按模块的完整对照见 [配置参考](#配置参考)。确定性所需的环境变量（NCCL / cuBLAS /
FlashAttention 等）已固化在 `tests/test_alignment_v4/common/{gcore,verl}/env.sh`，
按下文运行现成脚本即可，无需手动导出。

### 2. 准备数据

以 DAPO 数学 RL 为例：train = `zhuzilin/dapo-math-17k`，eval = `zhuzilin/aime-2024`。

- **gcore**：直接读原始 jsonl，无需转换（`data.py_path=tasks/math_rl_v4/dapo_dataset.py`）。
- **verl**：需先转成 parquet（各跑一遍）：

```bash
python3 tests/test_alignment_v4/common/verl/data/prepare_dapo_aime_data.py \
    --src hf-hub/zhuzilin/dapo-math-17k/dapo-math-17k.jsonl
python3 tests/test_alignment_v4/common/verl/data/prepare_dapo_aime_data.py \
    --src hf-hub/zhuzilin/aime-2024/aime-2024.jsonl
```

两侧数据必须满足的对齐约束：同一 `system_prompt`（逐字）、user 原文不改、
`enable_thinking=False`、`shuffle=False`、prompt/response 长度预算 2048/8192、reward
判分口径一致（`mathruler` 抽 `\boxed{}` → `score = acc + fmt`；verl 侧
[dapo_reward.py](../../tests/test_alignment_v4/common/verl/rewards/dapo_reward.py)
与 gcore `bt_reward_dapo.py` 1:1 一致）。

### 3. 运行

从仓库根目录（`cd /work/wepsdl/gcore-xxx`）执行现成脚本：

```bash
# gcore（默认 step1：on-policy + GRPO + dense + sglang）
bash tests/test_alignment_v4/rl/gcore/scripts/run.sh

# verl（对照）
LAUNCHER=mpirun bash tests/test_alignment_v4/rl/verl/scripts/run.sh
```

## 配置参考

按模块列出对齐所需的关键参数（gcore ↔ verl 对照）。除特别说明外，这些值都是
**对齐所必需**的显式设置，而非默认值。

### env（确定性 + 启动）

固化在 `common/{gcore,verl}/env.sh`，一般无需手动改。

| 环境变量 | 值 | 说明 |
|----------|-----|------|
| `GCORE_VERL_DETERMINISTIC` | `1` | verl 侧触发 torch 确定性 + 卸载 FA3（见 How It Works） |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | `0` | TransformerEngine 禁用非确定性算子 |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | cuBLAS 确定性 workspace |
| `NCCL_DETERMINISTIC` / `NCCL_ALGO` | `1` / `Ring` | NCCL 确定性、固定通信算法 |
| `FLASH_ATTENTION_DETERMINISTIC` | `1` | FlashAttention 确定性 |
| `PYTHONHASHSEED` | `42` | 冻结 hash 顺序（须在进程启动前设置） |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `1` | Megatron 要求 |

### system（确定性开关）

| 项 | gcore | verl |
|----|-------|------|
| 训练确定性 | `training.apply_deterministic_mode=True` | `override_transformer_config.deterministic_mode=True` |
| 训练 attention 后端 | `training.attention_backend=flash` | `override_transformer_config.attention_backend=flash` |
| 从 HF base 起跑 | `auto_load_from_save_ckpt=False`、`load_ckpt_path=null` | `trainer.resume_mode=disable`、`save_freq=-1` |

### data

| 项 | gcore | verl |
|----|-------|------|
| 样本顺序 | `data.shuffle=False` | `data.shuffle=False` |
| chat template | `enable_thinking=False`（默认） | `data.apply_chat_template_kwargs.enable_thinking=False` |
| prompt/response 长度 | `seq_length=10240`、rollout generate 8192 | `max_prompt_length=2048`、`max_response_length=8192` |
| system prompt | `data.system_prompt` | prepare 时写入 parquet，逐字相同 |

### infer_backend（SGLang rollout）

| 项 | gcore | verl |
|----|-------|------|
| 确定性推理 | 由 `apply_deterministic_mode` 触发 | `engine_kwargs.sglang.enable_deterministic_inference=True` |
| attention 后端 | `flashinfer` | `flashinfer` |
| cuda graph | `disable_cuda_graph=False` | `enforce_eager=False` |
| 并发上限 | `max_running_requests=128` | `max_num_seqs=128`（verl 默认 1024，必须显式设） |
| 采样 | `temperature=1.0`、`top_p=1.0` | 同 |

### training_backend（Megatron / mcore）

| 项 | gcore | verl |
|----|-------|------|
| 并行度 | `policy.dist_config` TP/PP/CP | actor/ref megatron TP/PP/CP |
| micro batch | `train_mbs=1` | `ppo_micro_batch_size_per_gpu=1` |
| 权重来源 | `build_from_mbridge=True` | `megatron.use_mbridge=True` |
| 序列 packing | `ppo_pack_seq=True`（THD） | `use_remove_padding=True`（THD） |
| recompute | `full / uniform / 1` | 同（`override_transformer_config`） |
| 优化器 | lr `1e-6` constant、wd `0.01` | 同 |

### loss（GRPO）

| 项 | gcore | verl |
|----|-------|------|
| advantage | `advantage_type=grpo`、`grpo_advantage_epsilon=1e-6` | `adv_estimator=grpo` |
| clip | `ppo_ratio_eps=0.2`、`ppo_dual_clip_ratio_c=20` | `clip_ratio=0.2`、`clip_ratio_c=20` |
| KL / entropy | `grpo_kl_loss_beta=0`、`ppo_entropy_bonus=0` | `use_kl_loss=False`、`entropy_coeff=0` |
| loss 归一化 | `calculate_per_token_loss=True` | `loss_agg_mode=token-mean` |

## How It Works

确定性与对齐分五层保证，每一层都要成立，端到端才能对上。

### 1. 确定性基础

两侧都需要在进程/通信初始化前固定 kernel 与 RNG 的确定性：torch 确定性算法、cuBLAS
workspace、NCCL 算法、FlashAttention 确定性、以及 `PYTHONHASHSEED`。gcore 由
`training.apply_deterministic_mode` 统一开启（同时强制禁用不稳定的 FA3），verl 侧由
`GCORE_VERL_DETERMINISTIC=1` 触发一个 `sitecustomize.py` 引导：在解释器启动最早期启用
torch 确定性并卸载 FA3。这层保证单框架内「同配置跑两遍曲线一致」。

### 2. 数据 / prompt 一致

对齐的前提是两侧喂进模型的 token 序列一致：同一 `system_prompt`、同样的 user 原文、
`enable_thinking=False` 的 chat template，以及 `shuffle=False` 固定样本顺序。reward 判分
两侧使用同一套规则（`mathruler` 抽 `\boxed{}`，`score = acc + fmt`），保证 advantage
的来源一致。

### 3. rollout 采样一致

SGLang 开启 `enable_deterministic_inference` 后，输出不再依赖同一 batch 里还有哪些请求；
每条请求再注入 `seed = 42 + 组内序号`，使采样可复现。两侧还需对齐并发上限
（`max_num_seqs=128`）与 attention 后端（flashinfer）等影响 batch 组成的参数。verl 侧
的采样 seed 修复（见 [对齐对 verl 的改动](#对齐对-verl-的改动)）保证 GRPO 组即使跨
worker 分片也不会复用同一 seed，从而避免同组样本塌缩、advantage 归零。

### 4. 训练前向 / logp 一致

训练侧两边都走 **THD 序列 packing**（gcore `ppo_pack_seq=True` / verl
`use_remove_padding=True`，共用同一套 `preprocess_packed_seqs`），使 packed 输入逐比特
一致；packed 前向的 `position_ids` 交给 mcore 按 `cu_seqlens` 重算（不外传
pre-pack 的 arange），使 logits 一致。严格 on-policy 时，还要保证 prev/ref 的 logp 与
训练前向走**同一条 THD 路径**，这样 importance ratio 恒为 1、梯度不被虚假 ratio 扰动。
最后，梯度按全局 response token 数归一（gcore `calculate_per_token_loss=True` ⟺ verl
`loss_agg_mode=token-mean`），保证 loss 口径一致。

### 5. 指标对比

gcore 与 verl 指标命名不同，`common/tools/metrics_map.yaml` 给出 gcore→verl 的映射；
gcore 上报时可按该映射额外翻译一份（`report.verl_metric_map_path`）。需要定位前向数值
发散时，两侧可落 dump（`GCORE_DUMP_*` / `VERL_DUMP_*`），用 `common/tools/` 下的
比对脚本逐层核对。

## 对齐对 verl 的改动

为了与 gcore 对齐，`verl` 仓库（`/work/wepsdl/projects/verl`）打了三处非 dump 的补丁：

- **`agent_loop.py`（采样 seed 对齐）**：verl在打开rollout确定性计算时，所有采样seed相同，
导致GRPO group 内所有 sample 全部相同，无法训练，同组样本塌缩、advantage 归零。patch 给
verl 的 sample 按照与 gcore 对齐的规则 assign 了 seed，使 verl 开启确定性时可以正常训练。

## 验证对齐

### gcore 自身可复现

同一份配置跑两遍，用 wandb 叠加两条曲线，确认逐 step 一致。

## 限制

- **跨框架 bitwise 不现实**：训练侧 bf16 数值噪声会随 step 级联放大，跨框架长期逐比特
  一致不可期。对齐目标是**曲线趋势一致**，而非无限步 bitwise，当然可以bitwise的话要尽量。
- **step 轴偏移**：gcore 的 step 从 0 计，verl 从 1 计，对比时以数值对齐、不要裸比
  step index（对齐点为 gcore step1 ↔ verl step2）。
- **确定性有性能代价**：确定性 kernel 更慢、部分算子 fallback，吞吐会下降；仅建议在
  调试 / 回归 / 研究对拍时开启，正式训练可关闭。

## Roadmap

后续需要对齐的方向：

- **后端（Backends）**
  - rollout：SGLang（已）、vLLM（待）
  - 训练：Megatron/mcore（已）、FSDP（待）
- **范式与算法（Tasks）**
  - 范式：RL（已，GRPO）、SFT、OPD（on-policy distill）
  - 算法：GRPO（已）、GSPO、SAPO 等
- **模型类型（Model types）**：dense（已，Qwen3-0.6B）、MoE（如 Qwen3-30B-A3B）、
  multimodal
- **工程优化（Features）**：dynamic-cp（动态 context parallel）等——放开时需重新验证
  prev/train 前向是否仍走同一 packing 门控。

具体对齐 readmap & 实验 see [gcore alignment roadmap](https://git.woa.com/wepsdl/gcore-dev/issues/29)
