# Dynamic Context Parallel (Dynamic CP)

> **Audience:** human operators + AI agents implementing or extending Dynamic CP.
> **Abbreviation:** this doc uses **dyn_cp**. Elsewhere in the repo, **DCP** means PyTorch `torch.distributed.checkpoint`.

---

## TL;DR (for agents)

Dynamic CP replaces fixed Context Parallel with **per-sequence variable CP group sizes** inside a shared DP×CP resource pool. Samples are **scheduled → all-to-all rerouted → packed into THD microbatches** (`micro_batch_size=1`), then forwarded with Ring Attention + `PackedSeqParams`.

**To add a new training type**, implement in `extended_model/`:

1. `{task}_reroute_data_for_dynamic_cp(...)` — preprocess + schedule + pack (+ return `routing_info` if forward-only reverse is needed)
2. `{task}_train_with_dynamic_cp(...)` — CP shard one packed microbatch → `(batch_dict, fwd_kwargs)`

Then branch at the training entry (`mcore_engine` / `mixin`) on `dist_config.dynamic_context_parallel`.

**Critical invariants:**

- Pre-shift tokens in reroute (`input=tokens[:-1]`, `target=tokens[1:]`). Never `roll(-1)` on packed THD buffers.
- Split packed logprobs with **`cu_seqlens_padded` offsets**, truncate to **`cu_seqlens` original length** per sample.
- Packed microbatches live on **CPU** after reroute; H2D happens lazily in `*_train_with_dynamic_cp`.
- Forward logprob uses `pre_shifted=True`, `ignore_cp=True` (CP comm handled in attention).
- **RL train loss:** after curr logprobs, reconstruct CP-sharded model outputs and **response-pad** all loss operands to dense `[B, R_max]` (`cu_seqlens_padded=None`, `local_cp_size=1` into `ppo_loss` / legacy loss). Megatron per-token denom uses **local shard** token count, not the reconstructed group mask.
- Logprob reverse path: last PP stage only → `reverse_reroute_logprobs` → `BroadcastUtils.broadcast_object_within_pp`.

---

## 背景 / Background

在 RL / SFT 训练中，同一 batch 内序列长度差异很大。固定 CP 对所有序列使用相同切分粒度，短序列浪费算力、长序列分配不均。

**Dynamic CP** 按每条序列长度动态选择 CP 组大小：长序列分配更多 GPU 做 CP 切分，短序列用更少 GPU 或打包到同一 rank。在不改变总 GPU 数的前提下提高吞吐。

In RL/SFT training, sequence lengths within a batch vary drastically. Fixed CP wastes compute on short sequences. **Dynamic CP** assigns CP group size per sequence to improve GPU utilization without changing total GPU count.

---

## 架构总览 / Architecture Overview

训练与 logprob 共用 Step 1–2；之后分叉：

```text
Raw GBS samples
    |
    v
[1 Reroute] preprocess -> schedule -> all-to-all -> pack THD -> CPU offload
    |
    v
[2 Forward] train_with_dynamic_cp
            model inputs: CP-shard to 1 x T_local
            rollout fields: stay 1 x T_global
            Megatron forward_backward (mbs=1)
    |
    +-----> [3a Train loss]
    |         curr_*: reconstruct -> jagged -> B x R_max
    |         rollout_*: packed_to_response_padded -> B x R_max
    |         ppo_loss / legacy (cu_seqlens_padded=None)
    |
    +-----> [3b Logprob-only]
              reassemble to 1 x T_global, split by cu_seqlens_*
              reverse_reroute_logprobs
              broadcast within PP
```

```mermaid
graph TD
  A[Raw GBS samples] --> B[Reroute preprocess]
  B --> C[Schedule and all-to-all]
  C --> D[Pack THD microbatches]
  D --> E[CPU offload]
  E --> F[Forward CP-shard model inputs]
  F --> G[Megatron forward_backward]
  G --> H[Train loss response-pad]
  H --> I[policy loss on BxR_max]
  G --> J[Logprob reassemble and split]
  J --> K[reverse_reroute_logprobs]
  K --> L[broadcast within PP]
```

### 并行维度交互 / Parallelism

| 维度 | 训练 | logprob (forward-only) |
|------|------|------------------------|
| **DP×CP** | reroute all-to-all; attention 内 Ring CP | 同上 + `reverse_reroute_logprobs` 还原 |
| **TP** | reroute 时 pad 对齐；forward 后 tokens 须 `% tp_size == 0` | 同训练 |
| **PP** | Megatron `forward_backward_func` 原生支持 | last stage 收集 logprob → reverse → **PP broadcast** |
| **VPP** | ❌ `virtual_pipeline_model_parallel_size > 1` | ❌ 同上 |

---

## 代码地图 / Code Map

| 文件 | 职责 |
|------|------|
| `gpatch_v4/configs/dist_config.py` | `DistConfig`：`dynamic_context_parallel`, `max_seqlen_per_dp_cp_rank`, `max_seqlen_per_dp_cp_rank_fwd_only`, `dynamic_cp_scheduler_type` |
| `gpatch_v4/configs/config.py` | 要求 `training.attention_backend='flash'`；**SFT** `FinetuneConfig` 另要求 `calculate_per_token_loss=True` |
| `gpatch_v4/configs/policy_config.py` | dyn_cp 与 smart_pad / dynamic_mbs 等互斥校验 |
| `gpatch_v4/core/parallel_state.py` | 初始化 Megatron dynamic CP parallel groups |
| `gpatch_v4/utils/dynamic_cp_utils.py` | 调度 / reroute / pack / `reverse_reroute_logprobs` / **loss reconstruct + response-pad helpers** |
| `gpatch_v4/extended_model/llm.py` | LLM 参考实现：`sft_*` / `grpo_*` reroute + train（CP-shard 仅 model inputs） |
| `gpatch_v4/extended_model/qwen3_vl.py` | VLM 参考实现（含 vision 字段） |
| `gpatch_v4/extended_model/base.py` | `PrepareDataForward` 接口定义 |
| `gpatch_v4/training_backend/megatron_backend/mcore_engine.py` | `rl_train_actor` / `finetune_step` reroute 入口；**`compute_log_probs_dynamic_cp`** |
| `gpatch_v4/training_backend/megatron_backend/mixin.py` | `rl_forward_step`（`_rl_response_pad_dyn_cp_tensors`）、`get_logprob_output_only_func_dynamic_cp`、`compute_logprobs` |
| `gpatch_v4/training_backend/loss/ppo_loss.py` | 新 loss：只吃 `[B, S]`（response-pad 之后） |
| `gpatch_v4/actor/grpo_train_actor.py` | rollout 阶段选择 dynamic-CP logprob 路径 |

---

## 配置 / Configuration

```yaml
policy:
  dist_config:
    tensor_model_parallel_size: 2
    pipeline_model_parallel_size: 2
    context_parallel_size: 1              # 最大 CP 组大小 / max CP group size
    dynamic_context_parallel: True
    max_seqlen_per_dp_cp_rank: 4096       # 训练调度预算；开启 dyn_cp 时必填
    # max_seqlen_per_dp_cp_rank_fwd_only  # 可选；默认 = 4 × max_seqlen_per_dp_cp_rank
    min_dynamic_context_parallel_size: 1
    dynamic_cp_scheduler_type: default    # default | smart_padding

  override_transformer_config:
    calculate_per_token_loss: True        # SFT FinetuneConfig 在 dyn_cp 下必填；RL GRPO 常用 True

training:
  attention_backend: flash                # dyn_cp 必填，否则 grad_norm NaN
```

### 参数说明 / Parameter Reference

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `dynamic_context_parallel` | bool | `False` | 开启 Dynamic CP |
| `max_seqlen_per_dp_cp_rank` | int | `None` | 每个 DP×CP rank 的最大 token 数（**训练** reroute 预算）。开启 dyn_cp 时必填 |
| `max_seqlen_per_dp_cp_rank_fwd_only` | int | `2 × max_seqlen` | **仅 forward** 路径（logprob / eval）的 reroute 预算。在 `DistConfig.__post_init__` 中自动填充 |
| `min_dynamic_context_parallel_size` | int | `1` | 最小 CP 组大小 |
| `context_parallel_size` | int | `1` | DP×CP 资源池上限；dyn_cp 在 `[min_cp, cp_size × dp_size]` 范围内选择 |
| `dynamic_cp_scheduler_type` | str | `"default"` | `"default"`: 全局排序 + all-to-all；`"smart_padding"`: 本地调度，无数据通信（需 smart padding 数据集） |

### 额外前置条件 / Additional Prerequisites

- Megatron-LM 含 Dynamic CP 支持（PR #2924, #3405, #2000）
- `context_parallel_size >= 2`
- 后端：`training_backend: mcore`
- 不支持 CUDA Graphs
- SFT 不能与 `training.use_dynamic_mbs` 同开
- SFT 不能与 `policy.smart_pad_train` 同开

---

## 已支持 / Supported Training Types

### 任务类型 / Tasks

| 训练类型 | 状态 | reroute | train / forward | logprob |
|----------|------|---------|-----------------|---------|
| GRPO / RL（含 async） | ✅ | `rl_reroute_data_for_dynamic_cp` | `grpo_train_with_dynamic_cp` → response-pad → policy loss | `compute_log_probs_dynamic_cp` |
| SFT | ✅ | `sft_reroute_data_for_dynamic_cp` | `sft_train_with_dynamic_cp` | eval 同样 reroute 后 forward（无 reverse reroute） |
| On-Policy Distill | ✅ | `rl_reroute_data_for_dynamic_cp` | `opd_train_with_dynamic_cp`（内部同 GRPO train）→ response-pad | `DistillStudentActor` → 同 `compute_log_probs_dynamic_cp`；teacher 也可开 dyn_cp |
| Router Replay (R3) | ✅ WeLM | 同上 + `routed_experts` 专用 pack | packed THD replay（与 tokens 同 THD 索引） | policy logprob replay |
| DPO | 🔜 | 需新增 | 需新增 | — |
| Off-Policy Distill | 🔜 | 需新增 | 需新增 | — |
| Reward Model (`rm_bt`) | ❌ | — | mixin assert 拒绝 dyn_cp | — |

### 模型 / PrepareData 实现

| 实现 | SFT | RL / OPD | 备注 |
|------|-----|----------|------|
| `extended_model/llm.py` | ✅ | ✅ + `opd_train_with_dynamic_cp` | 纯文本参考实现 |
| `extended_model/qwen3_vl.py` | ✅ | ✅ + OPD | 多模态；含 vision 字段 |
| `extended_model/wemm_video.py` | ✅ | ✅ + OPD | 视频 / 音频字段 |
| `extended_model/welm_v4.py` | ✅ | ✅（含 R3 `routed_experts`）；OPD 继承 LLM 的 `opd_train_with_dynamic_cp` | WeLM |
| `extended_model/welm_omni_v4_5.py` | ✅ | — | 当前仅见 SFT dyn_cp 路径 |
| `extended_model/deepseek_v4.py` | ✅ | ✅ train | 继承 LLM reroute；train 为 **contiguous** CP 切分 |

### RL policy loss（train 侧，response-pad 之后）

Dyn-CP 训练在 `mixin._rl_response_pad_dyn_cp_tensors` 之后进入 loss，操作数为 dense `B x R_max`（`cu_seqlens_padded=None`）。

| `ppo.loss_func` | dyn_cp | 说明 |
|-----------------|--------|------|
| `grpo` | ✅ | 默认；新旧 loss 路径均可 |
| `cispo` | ✅ | 同 token-level `[B, S]` |
| `sapo` | ✅ | 同；seq-mean 语义靠 `calculate_per_token_loss` / agg |
| `gspo` | ✅ | seq-level ratio 在 `[B, S]` 上算；须 `calculate_per_token_loss=False` |
| `steer` | ❌ | `grpo_train_actor` 显式 assert：暂不支持 dyn_cp |
| `opd`（legacy 名） / OPD 用 `grpo`+distill advantage | ✅ | OPD 训练走 `opd_train_with_dynamic_cp` + 同上 response-pad |
| `fipo` 等未进新 registry 的 | ❓ | 视是否走 legacy + 是否已 response-pad |

参考入口脚本示例：`tasks/math_rl_v4/scripts/grpo.sh`、`tasks/multimodal_v4/on_policy_distill/script/on_policy_distill.sh`（含 dyn_cp yaml）。

---

## 数据流详解 / Data Flow Details

### 1. Reroute（所有 rank 同步执行）

**入口：**

- GRPO 训练：`mcore_engine.rl_train_actor` → `rl_reroute_data_for_dynamic_cp`
- GRPO logprob：`mcore_engine.compute_log_probs_dynamic_cp`（reroute **一次**，ref/prev 共用 packed data）
- SFT：`mcore_engine.finetune_step` / `eval_step` → `sft_reroute_data_for_dynamic_cp`

**每个 sample 预处理后应包含：**

| 字段 | 形状 | 说明 |
|------|------|------|
| `tokens`, `labels`, `loss_mask`, `position_ids` | `[padded_len]` | 已 pre-shift（RL 额外字段见下） |
| `original_seq_len` | scalar int32 | shift 后真实长度 |
| `padded_seq_len` | scalar int32 | pad 后长度 |
| RL: `dyn_cp_response_start`, `dyn_cp_response_length` | scalar int32 | next-token 坐标下的 response span；train loss response-pad 用 |
| RL: `advantages`, `prev_log_probs`, `ref_log_probs`, … | `[padded_len]` | 与 tokens 等长；放进 `packed_keys` 即可一起调度 |
| WeLM R3: `routed_experts` | `[padded_len, num_layers, topk]` int32 | 通信时临时展平为 1-D，THD pack 后恢复 3-D |

**调度器：**

- **`default`** (`dyn_cp_schedule_default`): Megatron `DefaultDynamicCPScheduler` → 全局长度排序 → key-wise all-to-all → `build_packed_microbatches_by_keys`。返回 **`routing_info`**（GRPO logprob 反向通信用）。
- **`smart_padding`** (`dyn_cp_schedule_smart_padding`): 仅 SFT；本地选 cp_size，单次 scalar all-reduce(MAX)，**无样本 all-to-all**。要求 GBS 内 smart padding 后长度接近。

**Packed microbatch 输出字段（THD）：**

- `tokens`, `labels`, … — 1D concat，总长 `T_global` = 本 rank 该 microbatch 所有子序列 **padded** token 之和
- `cu_seqlens`, `cu_seqlens_padded` — 子序列边界（**logprob split 必须用 padded 偏移**）
- `max_seqlen`, `local_cp_size` — 本 microbatch 的 CP 组大小
- `dyn_cp_response_start` / `dyn_cp_response_length` — pack 后长度为 **本 MB 的 sample 数 `B`**（标量 cat，不是 `T_global`）
- `_dyn_cp_sample_ids` — 该 microbatch 内 global sample ID 列表（logprob 收集用）

**内存：** reroute 结束后 packed tensor **offload 到 CPU**；forward 时在 `*_train_with_dynamic_cp` 里 lazy `cuda(non_blocking=True)`。

**WeLM Router Replay：** 当前仅 WeLM v4/v4.5 支持
`dynamic_context_parallel=True + moe_router_replay=True`。每个 sample 的
`routed_experts` 先去掉 shifted RL input 不消费的末 token route，再按该
segment 的 padded 长度循环补齐。跨 DP×CP 通信时使用可逆的 1-D
flatten，pack 后恢复为 `[T, L, K]`。`grpo_train_with_dynamic_cp` 必须让
它与 tokens 共用同一个 `get_thd_partitioned_indices`，并保持 3-D；
router replay prepare 在该 THD shard 之后执行，只再处理 PP layer offset
和 TP sequence parallel，不能重复执行 fixed-CP 切分。

### 2. Train forward（每个 microbatch）

`*_train_with_dynamic_cp(batches, seqlen, ...)` 约定 **`len(batches) == 1`**。

步骤：

1. H2D（若 tensor 在 CPU）
2. **仅 model 输入** 在 `local_cp_size > 1` 时按 `get_thd_partitioned_indices(cu_seqlens_padded, ...)` 做 CP shard：`tokens` / `labels` / `position_ids`（VLM 另有 vision 相关输入）
3. **Rollout / loss 字段保持完整** `[1, T_global]`（组内 replicated，不切 CP）——与 verl 路径一致：只有带梯度的 model 输出才需要 reconstruct
4. Model 输入 `view(1, T_local)`；rollout 字段 `view(1, T_global)`
5. 构造 `PackedSeqParams(qkv_format="thd", local_cp_size=..., cp_group=...)`
6. `loss_mask` → `mask`，`labels` → `target`

| 类别 | 典型字段 | CP shard？ | Forward 后 shape |
|------|----------|------------|------------------|
| Model 输入 | `tokens`, `labels`→`target`, `position_ids` | ✅ | `[1, T_local]`，`T_local = T_global / local_cp_size` |
| Rollout / loss | `mask`, `advantages`, `prev_log_probs`, `ref_log_probs`, `rollout_log_probs`, `teacher_log_probs`, `prev_per_token_entropy`, `sample_mask` | ❌ 完整 | `[1, T_global]` |
| Meta | `cu_seqlens`, `cu_seqlens_padded`, `local_cp_size`, `max_seqlen` | — | 边界 / 标量 |
| Response span | `dyn_cp_response_start`, `dyn_cp_response_length` | — | 每 sample 一个标量，pack 后长度 = 本 MB sample 数 `B`（**不是** `T_global`） |

Logprob / CE 开关（shard 上算）：

```python
from_parallel_logits_to_logprobs(..., pre_shifted=True, ignore_cp=True)
# 或 logprobs_from_linear_ce(..., pre_shifted=True, ignore_cp=True)
```

### 2.1 RL train loss：response-pad 到 `[B, R_max]`

训练路径在算出 `curr_log_probs` / `per_token_entropy`（以及可选 `curr_topk_logprobs`）之后，**不在 THD shard 上直接算 policy loss**。由 `mixin._rl_response_pad_dyn_cp_tensors` 把所有 loss 操作数变成 dense `[B, R_max]`，再进 `ppo_loss` / legacy loss。

```
model 侧（CP-sharded，需保留本地 grad）:
  [1, T_local] ──reconstruct_dynamic_cp_packed_tensor──► [1, T_global]
             ──packed_to_jagged──► jagged [B, j1]
             ──jagged_to_response_padded──► [B, R_max]

rollout 侧（已完整 replicated）:
  [1, T_global] ──packed_to_response_padded──► [B, R_max]
```

| 字段 | 路径 | 进入 loss 前 shape |
|------|------|-------------------|
| `curr_log_probs`, `per_token_entropy`, `curr_topk_logprobs` | reconstruct → jagged → response-pad | `[B, R_max]` 或 `[B, R_max, K]` |
| `mask`, `advantages`, `prev_log_probs`, `ref_log_probs`, `rollout_log_probs`, `teacher_log_probs`, `prev_per_token_entropy`, `prev_topk_logprobs`, `token_weights` | `packed_to_response_padded` | 同上 |
| `sample_mask` | response-pad 后经 `as_sequence_sample_mask` | `[B]`（per-sample） |
| `cu_seqlens_padded` / `local_cp_size` 传入 loss | 强制 | `None` / `1` |

**Megatron 归一化：** `local_response_token_count` 在 reconstruct **之前**按本 rank THD shard 的 mask 计数（`dynamic_cp_local_packed_token_count`）。`calculate_per_token_loss=True` 时返回的 token 分母是这份 **local** count，不是 reconstructed `[B, R_max]` 的 `mask.sum()`（避免 DP×CP all-reduce 重复计 token）。

**显存：** 模型 activation 仍按 `T_local`；额外峰值主要是 loss 阶段若干份 logprob 量级的 `[1, T_global]` / `[B, R_max]`（远小于 hidden activation）。`local_cp_size==1` 时 reconstruct 为 identity。

### 2.2 Loss reconstruction helpers（`dynamic_cp_utils.py`）

| 函数 | 输入 | 输出 | 用途 |
|------|------|------|------|
| `reconstruct_dynamic_cp_packed_tensor` | CP shard `[1, T_local, ...]` + `cu_seqlens_padded` + `local_cp_size` | 完整 packed `[1, T_global, ...]` | 组内 all_gather；**本地 shard 保留 autograd**，远端 `detach`。`local_cp_size<=1` 直接返回 |
| `packed_to_jagged` | `[1, T_global, ...]` + `cu_seqlens_padded` + `cu_seqlens` | jagged `[B, j1, ...]` | 去掉每条序列 THD alignment pad，露出真实长度 |
| `jagged_to_response_padded` | jagged + `dyn_cp_response_{start,length}` | `[B, R_max, ...]` | 按 response span 右 pad；与下行 API 布局一致 |
| `packed_to_response_padded` | 完整 packed `[1, T_global, ...]` + 同上 span | `[B, R_max, ...]` | rollout 侧一步切 response；用显式 span，保留 mask 内部空洞 |
| `compute_dyn_cp_response_span` | `prompt_lengths` / `sequence_lengths`（或 fallback mask） | `(start, length)` | reroute 时写入 per-sample meta |
| `as_sequence_sample_mask` | response-pad 后的 `sample_mask` | `[B]` | pack 时 token-expand 的 sample mask 收成 per-sample |
| `dynamic_cp_local_packed_token_count` | 完整 packed mask + CP meta | scalar | Megatron per-token 分母用的 **本 rank** token 数 |

指标 reduce：scalar 在 dynamic CP 组内 AVG；token 级 `[sum, count]` 在 DP+CP 组 all-reduce。

### 3. GRPO logprob（`compute_log_probs_dynamic_cp`）专用路径

与训练共用 reroute + `grpo_train_with_dynamic_cp`，但走 **forward-only**，**不做** train 的 `[B, R_max]` response-pad。目标是还原成 **按 global sample id 的 1D logprob**，再 reverse 回 rollout owner。

```
grpo_train_actor.rollout
  └─ compute_log_probs_dynamic_cp          # mcore_engine.py
       ├─ rl_reroute_data_for_dynamic_cp(max_seqlen= max_seqlen_per_dp_cp_rank_fwd_only)
       ├─ _forward_packed_batches_unified  # Megatron forward_backward_func, micro_batch_size=1
       │    └─ get_logprob_output_only_func_dynamic_cp  # mixin.py
       └─ _reverse_and_collect            # per ref / prev
            ├─ [last PP only] reverse_reroute_logprobs  # single all_to_all_single
            └─ BroadcastUtils.broadcast_object_within_pp(result)
```

#### Logprob 阶段字段处理

| 阶段 | 字段 | Shape / 行为 |
|------|------|----------------|
| Reroute pack | `tokens`, `labels`, `position_ids` + 可选 RL 字段 | 同训练：THD concat；logprob-only 时 `mask` / advantages 等可能缺失 |
| `grpo_train_with_dynamic_cp` | Model 输入 CP-shard；若有 rollout 字段则仍完整 | Model：`[1, T_local]`；rollout（若有）：`[1, T_global]` |
| 算 logprob | `target`（sharded labels） | 输出 `logprobs`：`[1, T_local]` |
| CP 重组 | 同上 logprobs | → `[1, T_global]`：zigzag 用 scatter + all_reduce；contiguous 用 all_gather + cat |
| Per-sample split | 用 `cu_seqlens_padded` 偏移 + `cu_seqlens` 长度 | `List[Tensor]`，每条长度 = 该 sample `original_seq_len` |
| Reverse | `Dict[gid, Tensor]` | `reverse_reroute_logprobs` 发回原始 DP×CP owner（含全部 CP 兄弟） |

**`get_logprob_output_only_func_dynamic_cp` 内 per-sample split（易错点）：**

```python
# 偏移用 cu_seqlens_padded；长度用 cu_seqlens 原始长度
pad_start = cu_seqlens_padded[s_idx].item()
orig_len = cu_seqlens[s_idx + 1].item() - cu_seqlens[s_idx].item()
results.append(logprobs_flat[pad_start:pad_start + orig_len])
```

**与 train loss 的差异：**

| | Train loss | Logprob-only |
|--|------------|--------------|
| Model logprob 重组 | `reconstruct_dynamic_cp_packed_tensor`（本地留 grad） | scatter/all_gather（forward-only，无 backward） |
| 最终布局 | dense `[B, R_max]` response | per-sample 1D（含 prompt 段上的 next-token 位）后 reverse |
| 是否切 response | ✅ `dyn_cp_response_*` | ❌（reverse 后再由上游对齐 / mask） |

**与非 dynamic-CP 对齐：** 非 dyn_cp topk 路径在 `mixin.compute_logprobs` 做 `broadcast_object_within_pp`；dyn_cp 等价逻辑在 `_reverse_and_collect` 末尾（reverse reroute **之后**）。

### 4. `routing_info` 字段（GRPO logprob reverse 用）

由 `dyn_cp_schedule_default` 返回，经 `rl_reroute_data_for_dynamic_cp` 透传：

| key | 用途 |
|-----|------|
| `global_ids_this_rank` | 本 rank **reroute 前**拥有的 global sample ID |
| `global_id_logprob_lens` | `[(gid, logprob_len), ...]` 全局列表；len = `original_seq_len` |
| `gid_to_compute_rank` | 该 sample 在哪个 DCP rank 上算了 logprob |
| `gid_to_orig_dcp_rank` | 该 sample reroute 前属于哪些 DCP rank（该 DP index 下**所有** CP 兄弟 rank 的 list，而非单个 rank） |

`reverse_reroute_logprobs` 利用上述信息，**一次** `all_to_all_single` 把 logprob 发回原始 owner rank。

> **注意（2026.07 fix）**：rollout 数据在同一 DP index 的所有 CP 兄弟 rank 间是完全复制的
> （见 `is_mp_and_cp_head` + `broadcast_object_within_mp_and_cp`），所以每个 CP 兄弟 rank
> 都需要独立拿到一份 reverse 后的 logprob。`gid_to_orig_dcp_rank` 若只解析成单个
> rank（比如只用调用者自己 CP 切片内的 `dp_group` 反查），在 `cp_size>1` 时会因为
> 不同 CP 切片各自独立、自指地计算出**不同**的单一 owner，导致 `all_to_all_single`
> 的 send/recv split size 在跨 CP-parity 的方向上永远错配（一侧声明发 0，另一侧却
> 期望非 0），进而卡死。现已改为 `gid_to_orig_dcp_rank[gid]` 返回该 gid 所在 DP index
> 下的全部 CP 兄弟 rank，`reverse_reroute_logprobs` 对每个目标 rank 都重复发送一份。

---

## 接入新训练类型 / Adding a New Training Type

SFT / GRPO 遵循同一范式。按下列 checklist 实现（AI agent 可直接对照）。

### Step 0: 确认约束

- [ ] 不需要 VPP、MTP (deepseek_v3)；SFT dump + dyn-CP / `ppo_dump_moe_topk` 仍不支持
- [ ] `attention_backend=flash`；SFT 另需 `calculate_per_token_loss=True`
- [ ] 模型 extended class 继承 `PrepareDataForward`

### Step 1: `{task}_reroute_data_for_dynamic_cp`

**签名（GRPO 带 routing_info）：**

```python
def rl_reroute_data_for_dynamic_cp(
    self,
    gbs_batches: List[Dict[str, Any]],
    pad_token_id: int,
    pad_with_random_token: bool = False,
    **kwargs,  # vocab_size, max_seqlen_per_dp_cp_rank (override)
) -> Tuple[List[Dict], int, float, float, Dict[str, Any]]:
```

**SFT 无 routing_info：**

```python
) -> Tuple[List[Dict], int, float, float]:
```

**必须做：**

1. 计算 `pad_div = (2 * dp_cp_size) * tp_size`（smart_padding 见 `llm.py`）
2. 每条 sample：pre-shift、pad、写入 per-token 字段 + `original_seq_len` / `padded_seq_len`；**RL 还须** `dyn_cp_response_start` / `dyn_cp_response_length`（`compute_dyn_cp_response_span`）
3. 调用 `dyn_cp_schedule_default` 或 `dyn_cp_schedule_smart_padding`
4. 列出 `packed_keys`（所有需要一起 pack 的 per-token 字段 + response span meta）
5. GRPO：返回 `routing_info`；SFT smart_padding：不需要

**返回：**

- `packed_microbatches`: `List[Dict[str, Tensor]]`，CPU tensor
- `num_micro_batches`: forward 循环次数
- `seqlen_sum`, `seqlen_sq_sum`: 监控指标
- `routing_info`（可选）：logprob reverse 用

### Step 2: `{task}_train_with_dynamic_cp`

**签名：**

```python
def xxx_train_with_dynamic_cp(
    self,
    batches: List[Dict[str, Any]],  # len == 1
    seqlen: int,
    pad_token_id: int,
    ...
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    # returns (batch_dict_for_loss, fwd_kwargs_for_model)
```

**必须做：**

1. Lazy H2D
2. **仅** CP shard model 输入（`tokens` / `labels` / `position_ids`；用 `get_thd_partitioned_indices` + `get_dynamic_data_context_parallel_groups(group_size=local_cp_size)`）；rollout/loss 字段保持 `[1, T_global]`
3. TP 对齐 assert
4. `PackedSeqParams` + model 输入 `view(1, T_local)`；rollout 字段 `view(1, T_global)`
5. 字段 rename：`loss_mask→mask`, `labels→target`
6. RL：保留 `dyn_cp_response_*` 供 `mixin._rl_response_pad_dyn_cp_tensors` 使用

### Step 3: 训练入口分流

| 场景 | 文件 | 分支位置 |
|------|------|----------|
| RL 训练 reroute | `mcore_engine.rl_train_actor` | `dynamic_context_parallel` → `rl_reroute_data_for_dynamic_cp` |
| RL forward step | `mixin.rl_forward_step` | → `grpo_train_with_dynamic_cp` |
| RL logprob | `grpo_train_actor` + `mcore_engine.compute_log_probs_dynamic_cp` | 独立路径 |
| SFT reroute | `mcore_engine.finetune_step` / `eval_step` | → `sft_reroute_data_for_dynamic_cp` |
| SFT forward | `mixin._finetune_func` | → `sft_train_with_dynamic_cp` |

### Step 4（可选）: Forward-only / logprob

若新任务需要在 dyn_cp 下算 per-token 输出并还原到原始 sample 布局：

1. reroute 必须走 `dyn_cp_schedule_default` 并保存 `routing_info`
2. 实现或复用 `get_logprob_output_only_func_dynamic_cp` 模式的 forward step
3. last PP stage 按 `_dyn_cp_sample_ids` 收集 `Dict[gid, Tensor]`
4. 调用 `reverse_reroute_logprobs` + `broadcast_object_within_pp`
5. logprob reroute 预算用 `max_seqlen_per_dp_cp_rank_fwd_only`

### Step 5: 注册接口

在 `extended_model/base.py` 添加/更新抽象方法；在具体模型类 `@override`。

---

## 已知限制 / Known Limitations

| 不兼容项 | 行为 |
|----------|------|
| `virtual_pipeline_model_parallel_size > 1` | `DistConfig.__post_init__` assert |
| `model_arch == deepseek_v3` (MTP) | `_update_policy` assert |
| `ppo_dump_moe_topk > 0` + dyn-CP dump | `_update_policy` assert |
| SFT `ppo_dump_metrics_interval > 0` + dyn-CP | `_finetune_step` assert |
| `OffPolicyDistillConfig` | 未实现 |
| `ppo.loss_func='steer'` + dyn_cp | `grpo_train_actor` assert |
| Reward model training + dyn_cp | mixin assert |
| `training.use_dynamic_mbs` + dyn_cp SFT | `finetune_step` assert |
| `policy.smart_pad_train` + dyn_cp SFT | `_finetune_step` assert |
| GRPO + `smart_padding` scheduler | 未实现（GRPO / OPD 固定 `default`） |
| CUDA Graphs | 不支持 |

**已支持（相对早期文档）：**

- **PP > 1** 训练与 logprob（logprob 在 `_reverse_and_collect` 末尾 PP broadcast）
- **`max_seqlen_per_dp_cp_rank_fwd_only`** 独立 logprob 调度预算
- **RL train loss response-pad**：`reconstruct_*` / `packed_to_response_padded` → `[B, R_max]` 再进 `ppo_loss` / legacy
- **GRPO dump metrics**：loss 1D 字段 scatter 到 `[T-1]` 后 `reverse_reroute_logprobs` 回原始 DP
- **多模态** qwen3_vl / wemm_video；**WeLM R3**；**DeepSeek-V4** contiguous CP train

---

## 调参建议 / Tuning Tips

1. **`max_seqlen_per_dp_cp_rank`** — 最关键。太小 → microbatch 过多；太大 → OOM。建议从单卡最大 seq len 的 ~80% 起试。
2. **`max_seqlen_per_dp_cp_rank_fwd_only`** — logprob 无 backward，默认 2× 训练预算；OOM 时可单独调小。
3. **`min_dynamic_context_parallel_size`** — SFT 通常保持 `1`。
4. 观察 `[TRAIN-DYN-CP]` / `[SFT-DYN-CP]` 日志，`scheduled num_micro_batches` 越少通常越好。
5. 监控 `policy/dyn_cp_num_micro_batches`、`finetune/dyn_cp_num_micro_batches`。

---

## 示例 / Examples

```bash
# GRPO（默认开启 dyn_cp）
bash tasks/math_rl_v4/scripts/grpo.sh

# SFT dyn_cp
bash tasks/math_rl_v4/scripts/sft_dyn_cp.sh
```

关闭：`dynamic_context_parallel: False` → 回退固定 CP / BSH 路径。

---

## 正确性对齐实验 / Correctness Benchmarks

### math (qwen2)

| 模式 | 脚本 | acc | fmt |
|------|------|-----|-----|
| 原版 | `tasks/math_rl_v4/scripts/sft.sh` | 0.575 | 0.695 |
| dynamic-cp | `tasks/math_rl_v4/scripts/sft_dyn_cp.sh` | 0.578 | 0.697 |

WandB: [原版](http://wandb.testsite.woa.com:8080/plt2/dynamic_cp_riona/runs/0byzv68x) / [dyn_cp](http://wandb.testsite.woa.com:8080/plt2/dynamic_cp_riona/runs/s8qmm7rk)

### qwen3vl

| 模式 | 脚本 | acc |
|------|------|-----|
| 原版 | `tasks/multimodal_v4/finetune/scripts/finetune_qwen3vl_pmc_vqa.sh` | 0.65 |
| dynamic-cp | `tasks/multimodal_v4/finetune/scripts/finetune_qwen3vl_dyn_cp.sh` | 0.65 |

WandB: [原版](http://wandb.testsite.woa.com:8080/plt2/qwen3_vl_v4/runs/ptn8fb2m) / [dyn_cp](http://wandb.testsite.woa.com:8080/plt2/qwen3_vl_v4/runs/i9kml6ew)

---

## 调试清单 / Debug Checklist

logprob 数值异常（`ppo_ratio` / `k3_kl` 爆炸）或 train loss 异常时按序检查：

1. packed logprob split 是否用 **`cu_seqlens_padded`** 偏移 + **`cu_seqlens`** 原始长度
2. reroute 是否 pre-shift（禁止 post-pack `roll(-1)`）
3. ref/prev 是否共用**同一次** reroute 的 packed data
4. `reverse_reroute_logprobs` 的 `gid_to_*` 映射是否与调度一致
5. PP > 1 时是否在 reverse 之后做了 **`broadcast_object_within_pp`**
6. 训练 vs logprob 的 `max_seqlen_per_dp_cp_rank` 预算是否混用
7. train loss：batch 是否带 `dyn_cp_response_start` / `length`；进 loss 前是否已 response-pad（`cu_seqlens_padded is None`）
8. train loss：Megatron token 分母是否用 **local shard** count，而非 reconstructed `mask.sum()`
9. **reported `policy/loss` spikes only when `dyn_cp_local_cp_max>1`**：metrics all_reduce 前是否按 `local_cp_size` 除掉协作者重复（见 `_update_policy`）；未除时 CP 扩展的长序列 MB 会被计 k 次，均值被偏置。反传用 `n_alive_global` 不受影响，故 `grad_norm` 可能看起来正常。
9. `grpo_train_with_dynamic_cp`：是否只 CP-shard 了 model 输入，rollout 字段仍为 `[1, T_global]`

可选工具：`debug.compare_logprob_paths=True`（BSH vs dyn_cp 数值对比，见 `gpatch_v4/utils/logprob_compare_utils.py`）。
### 注意事项 / Notes

- packed 后每个 microbatch 在 token 维度上是 **THD 格式**（所有 sample 拼接成一条），所以 `micro_batch_size=1`。
- Forward 内不做 static CP all-reduce（CP 通信由 attention 层处理）；scalar 报告指标在动态 CP 组内 AVG，token 级 `[sum, count]` 在 DP+CP 组内 all-reduce。
- THD pack 后不能整体 `roll(-1)`（会污染子序列边界），所以在 reroute 里**预先 shift**（input = `tokens[:-1]`、target = `tokens[1:]`），logprob 用 `from_parallel_logits_to_logprobs(..., pre_shifted=True)`。
- RL **train** policy loss 在 response-pad 后的 `[B, R_max]` 上计算；**logprob-only** 仍按 packed THD 重组后按 sample split + reverse，二者不要混用 helper。
