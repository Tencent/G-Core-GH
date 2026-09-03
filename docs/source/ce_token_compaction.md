# CE Token Compaction

## 功能概述

CE token compaction 是一项**可选**优化：在 output projection 和 CE 之前，先丢掉不计 loss 的 token（prompt、padding、非 response 位置等），只对真正参与 loss 的 token 做计算，算完再 scatter 回原来的 dense 布局。YAML 开关是 `ce_compaction`；SFT 和 GRPO 都既支持 fused Linear CE，也支持普通 vocab-parallel CE。

这样可以减少 kernel 处理的 token 数，以及相应的计算和 TP 通信。上层 loss、weights、CP 规约和调用接口保持不变。

**默认关闭。** 关闭时训练路径与原来一致。

SFT 直接用已有的 `loss_mask` 选出有效 token。GRPO 的 compact-CE mask 是一对 `build_grpo_compact_ce_mask*` 函数：

- `build_grpo_compact_ce_mask`：普通 / static CP 的 `compute_logps`。这时还没有 padded mask，从每条 sample 拼出 `[B, S-1]`。`train_step` 不调它，那时 `generate_ppo_data` 已经把 `batch["mask"]` 挂上了。
- `build_grpo_compact_ce_mask_dyn_cp`：Dynamic CP 的 `compute_logps` 和 `train_step`。hidden / target 已经是当前 rank 的 `[T_local, 1, H]` / `[1, T_local]`，但 mask 往往还是整条 packed 的 `[1, T]`。这个函数按同样的 zigzag 切成 `[1, T_local]`。`compute_logps` 时如果还没有 mask，会先按 `dyn_cp_response_start/length` 造一条再切；`train_step` 时 mask 已经在，只切。

前者从 per-sample 造 `[B, S-1]`；后者把 packed `[1, T]` 切成当前 rank 的 `[1, T_local]`。

## 配置参数

在 yaml 的 `training` 节下设置：

| 参数 | dtype | default | 说明 |
|------|------|---------|------|
| `use_linear_ce` | bool | False | 是否使用 fused Linear CE；关闭时 compaction 使用普通 vocab-parallel CE |
| `linear_ce_backend` | str | `"separate"` | fused Linear CE 的反向实现，一般保持默认即可 |
| `ce_compaction` | bool | False | 是否在 output projection 和 CE 前压缩无效 token |

开启 compaction 后，模型会返回 output projection 前的 hidden states。`use_linear_ce: true` 时只对有效 token 调 fused Linear CE；`use_linear_ce: false` 时先只为有效 token 生成 vocab-parallel logits，再走普通 CE。两种路径都不会物化全序列 logits。

Linear CE 和 compact CE 都需要 mbridge 在 output projection 前返回 hidden_states，因此必须保持 `build_from_mbridge: true`。

### 配置示例

SFT：

```yaml
training:
  training_backend: mcore
  use_linear_ce: true
  linear_ce_backend: separate
  ce_compaction: true
```

如果需要普通 CE kernel，则改为：

```yaml
training:
  training_backend: mcore
  use_linear_ce: false
  ce_compaction: true
```

普通 CE 路径若同时设置 `cross_entropy_loss_fusion: true`，`cross_entropy_fusion_impl` 需为 `native` 或 `te`，不能保留 Linear CE 的 `linear`。

GRPO 同样在 `training` 节打开 compaction，并保持 `ppo.loss_func: grpo`。需要普通 CE kernel 时同样设 `use_linear_ce: false`。

## 何时有收益

收益来自少算那些最终会被 mask 掉的 token。长序列、真正算 loss 的 token 占比低、词表较大时更明显；短序列、或大部分 token 都算 loss 时，gather/scatter 开销会把省下的计算吃掉，基本没加速。

参考数字来自 fused Linear CE 的构造 workload，不是所有任务或普通 CE 路径都能复现同样幅度：

- SFT 32k：耗时大约减少 10%。
- GRPO：构造的 32k 长序列、有效训练 token 比例约 5% 时，`train_step` 耗时大约减少 20%，端到端大约减少 15%。

比较适合的场景包括：

1. SFT：prompt 很长、真正算 loss 的回答很短，例如长指令后只监督几个判定 token。
2. 搜索类多轮 GRPO：整段轨迹很长，有效训练 token 占比低。

## 精度

SFT / GRPO 上 compact 路径与对应的 dense CE 路径数值对齐（loss、有效 token 的 log-prob / 梯度）。compaction 会改变 token 维和规约顺序，**不承诺与 dense 路径 bitwise 一致**。这不表示 compact 路径本身原则上不能做到可复现；但当前实现尚未验证该组合，因此配置会拒绝同时设置 `apply_deterministic_mode: true`。

## 支持范围

| 场景 | 状态 |
|------|------|
| SFT / GRPO，CP=1、static CP、Dynamic CP | 支持 |
| 纯文本 Qwen3 / Qwen3-MoE、旧版 WeLM / DeepSeek-V4 / Omni / WEMM | 不支持，被 `model_arch` 白名单拦住 |
| on-policy / off-policy distillation、DPO、T2I RL | 不支持 |
| `smart_pad_infer`、static `ppo_pack_seq` | GRPO 不支持 |
| FSDP2、非 `grpo` 的 RL loss | 不支持 |

当前 `model_arch` 白名单：`qwen3_vl` / `qwen3_vl_moe` / `qwen3_5` / `qwen3_5_moe` / `welmv4_moe`。Qwen3.6 复用 `qwen3_5*` arch；`welmv4_moe` 仅支持实际构建 `MBridgeWelmV45Model` 的 WeLM v4.5 checkpoint。

白名单模型的 mbridge forward 返回 `hidden_states`、本 rank 的 output `weight` 和 `output_layer`。当前纯文本 `qwen3` / `qwen3_moe` 以及 `welm_moe` 构建的是 Megatron `GPTModel` / `WelmV4Model`，没有这些输出，即使普通 CE kernel 本身可用，也不能在 output projection 前做 compaction。

SFT 的 `loss_func` 仅支持 `cross_entropy` 和 `square_averaging_cross_entropy`。

## 注意事项

- 需要完整 logits 的能力不能和 Linear CE / compact CE 一起开，例如 `dump_metrics_logprobs_topk > 0`、`im_end_metrics_enable`、`log_prob_top_k > 0`。
- GRPO 的 compaction 只加速训练侧 CE（ref / prev `compute_logps` 和 `train_step` 的 current log-prob）。sampler 产生的 `rollout_log_probs` 不走这条路径。
