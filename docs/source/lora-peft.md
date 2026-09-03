# LoRA / PEFT 子系统

> 上次更新: 2026-08-16  
> 关联对话: [LoRA ckpt 集成与测试](d262bbf1-c5cb-4a97-829a-7a13c22cf232), [Per-expert LoRA 实现](1582425f-f2f5-46d5-9f37-3e5ef58e5f34)

## 一、概述

gcore-dev 支持两种 LoRA 后端：

| 后端 | 配置项 | 核心库 | 状态 |
|------|--------|--------|------|
| **mbridge** | `training.build_from_mbridge: True` | `mbridge/mbridge/peft/` | 主力，推荐 |
| **Megatron-Bridge** | `training.build_from_mbridge: False` | `Megatron-Bridge/src/megatron/bridge/peft/` | 旧版，仍可用 |

两者都通过 Megatron-Core 的并行线性层实现 LoRA adapter，但 checkpoint 保存/加载路径不同。

---

## 二、关键文件索引（按优先级）

### 必读

| 文件 | 内容 | 行数 |
|------|------|------|
| `gcore-dev/gpatch_v4/configs/lora_config.py` | `LoRAConfig` dataclass，所有 LoRA 超参定义 | ~70 |
| `gcore-dev/gpatch_v4/training_backend/megatron_backend/checkpoint.py` | checkpoint 保存/加载入口，HF 导出逻辑 | ~1100 |
| `mbridge/mbridge/peft/lora.py` | `LoRA`/`LoRAMerge`/`lora_merged`/`gather_lora_state_dict` | ~1200 |
| `mbridge/mbridge/peft/lora_layers.py` | `LoRALinear`/`LoRAGroupedLinear`/`LoRATopKRouter` | ~250 |
| `mbridge/mbridge/peft/canonical_lora.py` | `CanonicalLoRA`：自动处理 split QKV/FC1 和 per-expert | ~400 |

### 按需读

| 文件 | 何时需要 |
|------|----------|
| `mbridge/mbridge/peft/utils.py` | 修改 adapter 的并行策略、TP 分片方式 |
| `mbridge/mbridge/peft/base.py` | 理解 `PEFT` 基类 `walk_modules` 遍历 |
| `mbridge/mbridge/peft/walk_utils.py` | 理解模型遍历和 adapter 挂载流程 |
| `gcore-dev/gpatch_v4/training_backend/megatron_backend/mcore_peft.py` | `apply_peft_pre_wrap_hook`、`get_peft_cls`、`normalize_lora_config`、LoRA Coverage Report、`_verify_lora_weight_consistency` |
| `gcore-dev/gpatch_v4/training_backend/megatron_backend/mixin.py` | `MegatronBackendMixin` 中 PEFT 初始化流程（搜 `peft`） |

---

## 三、配置结构

```yaml
training:
  build_from_mbridge: True  # True=mbridge, False=Megatron-Bridge

policy:
  lora:
    rank: 128           # LoRA rank; 0 = 禁用 PEFT
    alpha: 256
    type: "canonical_lora"  # lora / vlm_lora / canonical_lora
    dropout: 0.0
    share_expert_adapters: false  # MoE: false=per-expert 独立，true=本地 experts 共享
    target_modules:       # 匹配规则见下文
      - "linear_q"        # canonical split targets
      - "linear_k"
      - "linear_v"
      - "*.in_proj_qkv"   # Qwen3.5 GDN split targets
      - "*.in_proj_z"
      - "*.in_proj_b"
      - "*.in_proj_a"
      - "*.linear_proj"
      - "*.linear_fc2"
      - "*.router"        # MoE router（可选）
```

配置 dataclass: `LoRAConfig` in `gpatch_v4/configs/lora_config.py`。

关键配置项补充：

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `check_lora_all_coverage` | `True` | 检查 LoRA 是否覆盖所有线性层（排除 output_layer/router） |
| `verify_weight_consistency` | `True` | 初始化时验证 LoRA 权重在 TP/DP/CP 维度的一致性 |
| `share_expert_adapters` | `False` | MoE routed experts 是否共享 adapter；默认每个 expert 独立 |

### 3.1 LoRA 类型对比

| 类型 | target_modules 写法 | 特点 |
|------|---------------------|------|
| `lora` | 直接写 fused 层名（`linear_qkv`, `linear_fc1`） | 对 fused 层整体做一个 adapter |
| `canonical_lora` | 写 split 子层名（`linear_q`, `linear_k`, `linear_fc1_gate`） | 自动拆分 fused 层为独立子 adapter |

---

## 四、核心机制

### 4.1 Adapter 挂载

`LoRA.transform()` / `CanonicalLoRA.transform()` 遍历模型，将匹配的线性层替换为对应的 wrapper：

| 原始层类型 | Wrapper | 说明 |
|-----------|---------|------|
| ColumnParallel / RowParallel | `LoRALinear` | 标准 LoRA |
| ColumnParallel (linear_qkv) | `LoRALinearSplitQKV` | canonical: Q/K/V 独立 adapter |
| ColumnParallel (linear_fc1) | `LoRALinearSplitFC1UpGate` | canonical: gate/up 独立 adapter |
| Qwen3.5 GDN (in_proj) | `LoRALinearSplitGDNInProj` | canonical: QKV/Z/B/A 独立 adapter |
| `TEGroupedLinear` (MoE experts) | `LoRALinear` + `GroupedExpertLinearAdapter` | **per-expert LoRA** |
| `TopKRouter` | `LoRATopKRouter` | Router adapter |
| `nn.Linear` (ViT/projector) | `LinearAdapter` | 非 Megatron 层 |

```
标准:
  LoRALinear
    ├── to_wrap: TEColumnParallelLinear (原始层)
    └── adapter: ParallelLinearAdapter
          ├── linear_in:  ColumnParallelLinear(in_features, dim)   → lora_A
          └── linear_out: ColumnParallelLinear(dim, out_features)  → lora_B

Per-expert:
  LoRALinear
    ├── to_wrap: TEGroupedLinear (原始层, 所有本地 experts)
    └── adapter: GroupedExpertLinearAdapter
          ├── linear_in.weight:  [local_experts, rank, in]
          └── linear_out.weight: [local_experts, out, rank]
```

### 4.2 Per-Expert LoRA（MoE 专用）

针对 `TEGroupedLinear`（MoE routed experts），每个专家拥有独立的 LoRA adapter：

- **创建**: 检测 grouped expert linear 且 `share_expert_adapters=False` 时创建 `GroupedExpertLinearAdapter`
- **前向**: base forward 仍走 grouped GEMM（性能不变），adapter 按 `tokens_per_expert` 逐专家 dispatch
- **Merge**: `LoRAMerge.transform()` 将 packed adapter 的每个 expert delta 合入对应 base expert
- **导出**: `gather_lora_state_dict()` 通过 EP group `all_gather` 收集所有 rank 的 adapter 权重
- **HF key**: `base_model.model.model.language_model.layers.X.mlp.experts.gate_up_proj.{expert_id}.lora_A.weight`

性能开销：adapter FLOPs ≈ rank/ffn_hidden ≈ 1%，可忽略。

`share_expert_adapters` 仅影响 MoE routed expert 的 grouped linear：
- `false`（默认、推荐）：每个 local expert 有独立 A/B，使用
  `GroupedExpertLinearAdapter` 的 grouped-mm 路径。
- `true`：同一 EP rank 上的 local experts 共享一个 `ParallelLinearAdapter`，
  参数更少，但会降低 expert-specific adapter 容量。

### 4.3 TP 分片（关键）

adapter 的 `linear_in`/`linear_out` 是 Megatron 并行线性层，TP > 1 时权重被分片：

| 基座层类型 | linear_in 类型 | linear_in shape | linear_out shape |
|-----------|---------------|-----------------|------------------|
| ColumnParallel (qkv, fc1) | ColumnParallel | `(dim/TP, in)` | `(out/TP, dim)` |
| RowParallel (proj, fc2) | RowParallel | `(dim, in/TP)` | `(out/TP, dim)` |

- **merged 导出**: `LoRAMerge.merge()` 内部做 `all_gather` 重建完整 delta → 正确
- **adapter 导出**: `gather_lora_state_dict()` 做 `all_gather` 重建完整权重 → 正确
- Column 层 gather dim 0, Row 层 gather dim 1
- **Per-expert adapter 不做 TP sharding**（直接 full-size），通过 EP group all_gather 收集

### 4.4 HF 名称映射

mcore → HF PEFT 的参数名转换在 `mbridge/mbridge/peft/lora.py`:

```python
# mcore_adapter_name_to_hf() 的映射:
# 标准层:
"self_attention.linear_qkv"  → "self_attn.qkv_proj"
"self_attention.linear_proj" → "self_attn.o_proj"
"mlp.linear_fc1"             → "mlp.gate_up_proj"
"mlp.linear_fc2"             → "mlp.down_proj"
"adapter.linear_in.weight"   → "lora_A.weight"
"adapter.linear_out.weight"  → "lora_B.weight"

# Canonical split:
"adapter.adapter_q.linear_in" → "q_proj.lora_A"
"adapter.adapter_gate.linear_in" → "gate_proj.lora_A"

# Per-expert (通过 bridge 映射):
"experts.linear_fc1.adapter.3.linear_in" → "experts.gate_up_proj.3.lora_A"
"experts.linear_fc2.adapter.3.linear_out" → "experts.down_proj.3.lora_B"
```

`gather_lora_state_dict()` 直接返回 HF 格式 key。

### 4.5 LoRA 权重一致性验证

`_verify_lora_weight_consistency()` 在 PEFT 注入后检查 LoRA 权重在各并行维度的正确性，通过 `policy.lora.verify_weight_consistency: True` 启用。

检查范围：所有 adapter 类型（与 `_log_lora_coverage` 一致的 9 种），只收集有 `linear_in` 属性的叶子 adapter。

| 并行维度 | 检查规则 |
|----------|----------|
| **TP** | replicated adapter（`linear_in` 是 `nn.Linear`）必须一致；TP-sharded adapter（`linear_in` 是 `ColumnParallelLinear`/`RowParallelLinear`）跳过（权重被刻意切分） |
| **DP** | 所有 adapter 必须一致（同一 TP shard 在不同 DP rank 上应相同） |
| **CP** | 所有 adapter 必须一致（同一 TP shard 在不同 CP rank 上应相同） |

额外检查：replicated adapter 的 LoRA 参数必须有 `average_gradients_across_tp_domain=True` 标记（parallel adapter 通过 Column/RowParallelLinear 内部处理梯度同步，不需要此标记）。

TP 切分判定：通过 `_is_tp_sharded(adapter)` 检查 `adapter.linear_in` 的类型，而非硬编码 adapter 类型名。

注意事项：
- BFloat16 参数需先 `.float()` 再转 numpy（否则 `TypeError: Got unsupported ScalarType BFloat16`）
- 使用 MD5 hash + allgather 通信模式，无需在大 world_size 下传输完整参数

### 4.6 Merged 导出（无数值漂移）

`lora_merged(models)` 上下文管理器：
1. 备份原始 base weight 到 pinned CPU memory（`tensor.mbridge_cpu_data`）
2. 原地 merge LoRA delta（含 TP all_gather / per-expert 逐个 merge）
3. 将 wrapper 替换为 `to_wrap`（消除 `.to_wrap.` 前缀使 `named_parameters()` 干净）
4. yield → bridge 保存
5. finally: 从 CPU pinned memory 恢复权重（零漂移），恢复 wrapper 模块

CPU backup buffer 通过 `tensor.mbridge_cpu_data` 属性缓存，重复 save 时复用同一 pinned allocation。

---

## 五、Checkpoint 保存/加载

### 5.1 HF 导出入口

`bridge_save_hf()` → 分发到:
- mbridge + PEFT: `_mbridge_save_hf_adapter()`
- Megatron-Bridge + PEFT: `_megatron_bridge_save_hf_adapter()`
- 无 PEFT: `_mbridge_save_hf()` / `_megatron_bridge_save_hf()`

导出产物：
- `hf/<step>/`: merged 完整模型（safetensors）
- `hf/<step>_adapter/`: adapter_model.safetensors + adapter_config.json

### 5.2 分布式 Checkpoint（断点续训）

mbridge 路径的特殊处理（`save_checkpoint` / `load_checkpoint` in `checkpoint.py`）:
- **保存**: 禁用 PEFT state dict 过滤（`build_from_mbridge=True` 时不调用 `_apply_peft_state_dict_filter`），保存完整 state
- **加载**: 通过 `_remap_peft_sharded_keys()` 将 `sharded_state_dict` 的扁平 key 映射回 `model.state_dict()` 的 `.to_wrap.` key；使用 `strict=False` 加载
- **注意**: 模型结构变化后需 `no_load_optim: True` 跳过旧 optimizer

### 5.3 `peft_to_run_config_dict()`

序列化 PEFT 配置写入 `run_config.yaml`。手动遍历 dataclass fields 避免 `dataclasses.asdict` 在 Python 3.10 下对 `torch.dtype` 的 `TypeError`。

---

## 六、测试方法

### 验证脚本

`gcore-dev/tests/test_gpatch_v4/test_lora_mbridge.py` — 自动检测模型类型并分发:

| 模型 | 检测条件 | 比较函数 |
|------|----------|----------|
| Qwen2 (dense) | model_type 非 qwen3_5/qwen3_vl | `compare_qwen2` |
| Qwen3-VL | model_type = qwen3_vl/qwen2_vl | `compare_qwen3_vl` |
| Qwen3.5 MoE (lora) | model_type = qwen3_5 + 无 separate qkv | `compare_qwen3_5` |
| Qwen3.5 MoE (canonical_lora) | model_type = qwen3_5 + 有 separate qkv | `compare_qwen3_5_moe` |

测试逻辑: `original_base + adapter_deltas == merged_export`（strict `torch.equal`）

处理的特殊情况：
- QKV interleaving（Qwen3.5 output_gate: Q,G,K,V 布局）
- Fused gate_up_proj 拆分（shared_expert）
- Per-expert adapter 对 3D stacked tensor 的 slice 应用
- MTP 模块移除（mtp_num_layers=null 时不导出）

### 已验证的测试矩阵

| 测试 | 模型 | 类型 | TP | EP | 验证内容 | 结果 |
|------|------|------|----|----|----------|------|
| adapter merge | Qwen2.5-Math-1.5B | lora | 1 | 1 | torch.equal | PASS |
| adapter merge | Qwen2.5-Math-1.5B | canonical_lora | 1 | 1 | torch.equal | PASS |
| adapter merge | Qwen3-VL-4B | lora | 1 | 1 | torch.equal | PASS |
| adapter merge | Qwen3.5-35B-A3B | canonical_lora | 1 | 8 | torch.equal | 待验证 |
| adapter merge | Qwen3.5-35B-A3B | lora | 1 | 8 | torch.equal | 待验证 |

### 测试运行

```bash
cd /mnt/ceph-hz1-csp/mm-base-plt2/user_guanyouhe/wepsdl/gcore-dev

# GDN fused in_proj 的 QKV/Z/B/A split-LoRA 单测
PYTHONPATH="$PWD:../Megatron-LM:../mbridge:${PYTHONPATH:-}" \
python -m pytest -q tests/test_gpatch_v4/test_gdn_split_lora.py

# 验证 adapter merge 正确性
python tests/test_gpatch_v4/test_lora_mbridge.py \
     --original hf-hub/Qwen/Qwen3.5-35B-A3B \
     --merged ckpt_qwen3_5_35b_a3b_pmc_vqa_full_lora/hf/10 \
     --adapter ckpt_qwen3_5_35b_a3b_pmc_vqa_full_lora/hf/10_adapter
```

---

## 七、与 HuggingFace PEFT 的对比

HF PEFT 也支持 MoE per-expert LoRA（v0.17.0+），对比：

| | HuggingFace PEFT | mbridge |
|--|--|--|
| 存储 | 3D `nn.Parameter` (batch) | `nn.ModuleList` (per-expert) |
| 目标 | `target_parameters`（3D nn.Parameter） | 自动检测 `TEGroupedLinear` |
| rank | 建议 `r // num_experts`（控制参数量） | 使用完整 rank |
| EP 支持 | 无（单节点/DDP） | 原生支持 EP all_gather |
| 推理 | 需 `merge_and_unload()` | 训练后直接 merge 导出 |

---

## 八、已知坑点

1. **Python 3.10 `dataclasses.asdict` + `torch.dtype`**: 会抛 `TypeError`，已用手动序列化绕过
2. **adapter 导出 TP>1**: 通过 `gather_lora_state_dict` 做 all_gather（已修复）
3. **adapter 导出 EP>1**: per-expert adapter 需跨 EP group all_gather 收集所有 rank 的本地专家（已修复）
4. **分布式 ckpt 的 key 不匹配**: `sharded_state_dict()` 扁平化了 `.to_wrap.`，用 `_remap_peft_sharded_keys()` 解决
5. **bridge fused expert 命名**: bridge 将 experts 导出为无 `.weight` 后缀的 3D 融合 key（如 `model.language_model.layers.X.mlp.experts.gate_up_proj`），adapter key 中需嵌入 expert_id
6. **`lora_A` 非零不代表模型被修改**: `lora_A` 随机初始化，`lora_B` 初始化为零。lr=0 时 delta = B@A = 0
7. **BFloat16 tensor 转 numpy**: `_param_hash` 需先 `.float()` 再 `.numpy()`，否则 `TypeError: Got unsupported ScalarType BFloat16`
8. **`lora` 类型在 fused 层（`linear_qkv` / `linear_fc1`）下，TP>1 与 TP=1 数学上不等价**：
    - 现象：`lora` 类型在 Qwen3.5-35B-A3B 上 TP=1 acc=0.7056，TP=2 acc=0.6891，差 ~1.65%；同模型 `canonical_lora` TP=1=0.7159、TP=2=0.7186 几乎一致。
    - 根因 1（fused QKV）：`LoRA.transform()` 对 `linear_qkv` 只挂一个 `ParallelLinearAdapter`，`linear_out` 是非 strided `ColumnParallelLinear`，TP>1 时 `B` 沿 `dim 0` 被切成 `(q+k+v)/TP` 的 contiguous chunk；而 base `linear_qkv` 是 strided/interleaved（按 `num_query_groups` 把 Q 多头与 K/V 交错）。前向数值正确（adapter_output 与 base output 同 layout 对齐相加），但这等价于在 `B` 矩阵上加了一个"按 contiguous TP 块切分"的隐式约束，**Q/K/V 之间无法在低秩瓶颈中共享信息**——TP=1 没有这个约束，所以训练轨迹和最终权重不同。
    - 根因 2（strided `linear_fc1`）：base 的 gate/up 是 stride=2 interleaved（`[r0_gate, r0_up, r1_gate, r1_up]`），adapter `linear_out` 是 contiguous chunk（`[r0_(前一半), r1_(后一半)]`）。前向数学正确（同通道相加+swiglu），但 `B` 的低秩容量被 contiguous 切分浪费，gate/up 跨 rank 无法共享。Merge 时通过 `_deinterleave_gathered_lora_b` 修正导出 layout，**但训练损失无法事后修复**。
    - `canonical_lora` 把 `linear_qkv` 拆成独立 `adapter_q/k/v`、`linear_fc1` 拆成 `adapter_gate/up`，每个子 adapter 有自己完整的 `A/B`，TP 切分发生在子 adapter 内部，不跨 fused 边界，因此 TP=1/2 等价。
    - **建议**：fused 层 + LoRA + TP>1 场景必须用 `canonical_lora`；如必须用 `lora` 类型，建议 TP=1（搭配 EP / DP 扩展）。

---

## 九、Demo 配置参考

### Qwen3.5 MoE (canonical_lora，推荐)

```yaml
policy:
  dist_config:
    tensor_model_parallel_size: 1
    expert_model_parallel_size: 8
  lora:
    rank: 128
    alpha: 256
    type: "canonical_lora"
    share_expert_adapters: false
    target_modules:
      - "linear_q"
      - "linear_k"
      - "linear_v"
      - "linear_fc1_up"
      - "linear_fc1_gate"
      - "*.linear_proj"
      - "*.in_proj_qkv"
      - "*.in_proj_z"
      - "*.in_proj_b"
      - "*.in_proj_a"
      - "*.out_proj"
      - "*.qkv"
      - "*.proj"
      - "*.linear_fc2"
      - "*.router"

checkpoint:
  no_save_optim: True
  no_load_optim: True        # 模型结构变化后必需
  convert_mcore_to_hf_online: True
```

脚本: `gcore-dev/tasks/multimodal_v4/finetune/scripts/finetune_qwen3_5_moe_lora.sh`

## 测试结果（add by gyhe）

### math
任务：Math SFT (Qwen)，详见 `gcore-dev/docs/source/math_sft.md`

测试脚本：`gcore-dev/tasks/math_rl_v4/scripts/sft_lora.sh`

| LoRA Type | LR | Eval Accuracy | Format Matching Degree | 备注 |
|-----------|------|--------------|----------------------|------|
| full SFT (baseline) | 2.0e-5 | 0.58 | 0.70 | 无 LoRA，全量微调 |
| mbridge lora | 2.0e-4 | 0.56 | 0.70 | lora 好像对初始化比较敏感 |
| mbridge canonical_lora | 2.0e-4 | 0.56 | 0.69 | lora 好像对初始化比较敏感 |


### qwen3_vl-4B(dense) VQA

测试脚本：`gcore-dev/tasks/multimodal_v4/finetune/scripts/finetune_qwen3vl_lora.sh`

| 配置 | TP | LR | Acc | 备注 |
|------|------|------|------|------|
| src (baseline) | - | - | 0.5 | 原始模型 |
| src (baseline) | 2 | 5.0e-7 | 0.65 | 全量微调 |
| canonical_lora | 1 | 5.0e-6 | 0.6658 | - |
| canonical_lora | 2 | 5.0e-6 | 0.6684 | - |
| lora | 1 | 5.0e-6 | 0.6643 | - |
| lora | 2 | 5.0e-6 | 0.6677 | - |


### qwen3.5-4B(dense) VQA

测试脚本：`gcore-dev/tasks/multimodal_v4/finetune/scripts/finetune_qwen3_5_dense_lora.sh`

| 配置 | TP | LR | Acc | 备注 |
|------|------|------|------|------|
| src (baseline) | 2 | 5.0e-7 | 0.4121 | 原始模型 |
| src (baseline) | 2 | 5.0e-7 | 0.6843 | 全量微调 |
| canonical_lora | 1 | 5.0e-6 | 0.6894 | - |
| canonical_lora | 2 | 5.0e-6 | 0.6839 | - |
| lora | 1 | 5.0e-6 | 0.6891 | - |
| lora | 2 | 5.0e-6 | 0.6831 | - |

### qwen3.5-35B-A3B(MoE) VQA

测试脚本：`gcore-dev/tasks/multimodal_v4/finetune/scripts/finetune_qwen3_5_moe_lora.sh`

| 配置 | TP | EP | LR | Acc | 备注 |
|------|------|------|------|------|------|
| src (baseline) | - | - | - | 0.5979 | 原始模型 |
| full SFT | 2 | 8 | 5.0e-7 | 0.7095 | 全量微调，free-router |
| canonical_lora | 1 | 8 | 5.0e-6 | 0.7159 | - |
| canonical_lora | 2 | 8 | 5.0e-6 | 0.7186 | - |
| lora | 2 | 8 | 5.0e-6 | 0.6891 | TP>1 fused 层有容量损失，见坑点 #8 |
| lora | 1 | 8 | 5.0e-6 | 0.7056 | 仍低于 canonical_lora，见坑点 #8 |
