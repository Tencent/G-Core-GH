# Reproducibility / 训练复现

本文介绍如何让 gcore 的 RL 训练做到**逐 bit 可复现**（bit-wise reproducible）：需要打开哪个开关、默认 seed 如何工作，以及复现必须保持不变的前置条件。

This doc explains how to make gcore RL training **bit-wise reproducible**: which switch to turn on, how the default seeds work, and which preconditions must stay unchanged across runs.

## TL;DR

只需要打开开关：

You only need to turn on:

```bash
+training.apply_deterministic_mode=True
```

可参考开箱即用的示例脚本 `tests/test_alignment_v4/rl/gcore/scripts/run.sh` 与配置 `tests/test_alignment_v4/rl/gcore/config/align.yaml`（同一份配置跑两遍即复现；MoE 用 `CONFIG_NAME=repro_moe`）。

See the ready-to-run example `tests/test_alignment_v4/rl/gcore/scripts/run.sh` and config `tests/test_alignment_v4/rl/gcore/config/align.yaml`.

## 需要打开的开关 / The switch to enable

`training.apply_deterministic_mode=True` 是唯一需要手动打开的开关。打开后 gcore 会在训练后端、推理后端、配置校验三处自动完成一系列确定性设置，**无需手动设置其他环境变量**。

`training.apply_deterministic_mode=True` is the only switch you need to set manually. Once enabled, gcore automatically applies deterministic settings across the training backend, the inference backend, and config validation — **no extra environment variables needed**.

它具体做的事情：

What it does under the hood:

1. **进程级环境变量**（在 NCCL 通信器创建前设置 / set before NCCL communicators are created）：

   ```bash
   NCCL_DETERMINISTIC=1            # 强制确定性集合通信
   NCCL_ALGO=Ring                 # 固定 Ring 归约算法
   FLASH_ATTENTION_DETERMINISTIC=1
   NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
   CUBLAS_WORKSPACE_CONFIG=:4096:8  # 固定 cuBLAS workspace，保证 GEMM 可复现
   ```

2. **Torch / cuDNN 层面**：`torch.use_deterministic_algorithms(True)`、`cudnn.deterministic=True`、`cudnn.benchmark=False`。

3. **Megatron / Bridge 模型**：向 transformer config 注入 `deterministic_mode=True`。

4. **推理后端（rollout 生成）**：
   - vLLM：设置 `VLLM_BATCH_INVARIANT=1`（batch 不变性，输出与同 batch 内其他请求无关）。
   - SGLang：传入 `enable_deterministic_inference=True`。

5. **配置校验与冲突处理**：
   - 禁止 `checkpoint.skip_save_mcore_model=True`（HF bridge 来回转换有精度损失，破坏断点续训的确定性）。
   - 若 `attention_backend=flash`，自动设 `GPATCH_DISABLE_FA3=1` 回退到 FA2（FA3 的确定性 backward 在 Hopper 上因全局 atomics 不可靠）。

## Seed：不需要手动设置 / Seeds: no manual setup needed

所有相关 seed 都有默认值（均为 42），示例 config 没有覆盖，因此**无需手动设置任何 seed**。

All relevant seeds default to 42, and the example config does not override them, so you **do not need to set any seed manually**.

- **训练 seed**：`training.seed`（默认 `42`）。初始化并行状态时自动应用到 `random` / `numpy` / `torch` / Megatron TP CUDA RNG tracker，并按 pipeline rank 偏移（`seed + 100 * pp_rank`）保证各 PP stage 不同。

- **Rollout 引擎 seed**：`sampler.infer_engine_configs[i].engine_seed`（默认 `42`），传给 SGLang `random_seed` / vLLM `seed`。

- **Rollout 采样 seed**：`sampler.infer_engine_configs[i].seed`（默认 `42`）。生成时每个 `prompt × repeat` 在该基准上加偏移 `i * sampling_repeat_n + j`，保证每条样本**确定但互不相同**。

> 即使 `temperature=1.0`（随机采样），因为有固定 seed + 偏移，rollout 结果仍然可复现。
>
> Even with `temperature=1.0` (stochastic sampling), rollout outputs stay reproducible thanks to the fixed seed + offset.

如需显式锁定，可在命令行或 yaml 覆盖：

To pin explicitly, override on the CLI or in yaml:

```bash
+training.seed=1234
```

## 复现的前置条件 / Preconditions for reproduction

确定性开关只保证“同样的输入 + 同样的运行环境 → 同样的输出”。要让两次运行结果一致，必须保持以下不变：

The deterministic switch only guarantees "same input + same runtime → same output". For two runs to match, the following must stay identical:

- **并行度**：TP / PP / CP / EP 等所有并行 size 不变。并行度会改变 RNG 分布和通信归约顺序，即使 seed 相同也无法逐 bit 复现。
- **Batch 配置**：`train_gbs` / `train_mbs` / `rollout_gbs` / `sampling_repeat_n` 等不变。
- **模型与数据**：相同的模型权重、tokenizer、数据集与顺序。
- **seed 一致**：两次用同一个 seed（默认都是 42 即一致）。
- **确定性开关常开**：两次都保持 `apply_deterministic_mode=True`。
- **写死 attention backend**：`training.attention_backend` 默认是 `auto`，TE 会**根据环境（GPU 架构、序列长度、dtype 等）自动挑选** attention 后端，不同机器/不同 batch 选到的 kernel 可能不一样，从而破坏复现。复现时务必显式指定，例如 `+training.attention_backend=flash`（确定性模式下 gcore 会自动把 FA3 回退到 FA2），不要用 `auto`。

## 断点续训的注意事项 / Notes on resume

- 复现场景下**不要**打开会丢弃 optimizer / RNG 状态的开关，断点续训需要完整的 optim 与 rng 才能精确续上（参见 [Resume Training](resume.md)）。
- 某些 checkpoint 选项（如 `skip_save_mcore_model`、在线 HF bridge 转换）会引入精度损失，使断点续训后的确定性失效；确定性模式下 gcore 已对 `skip_save_mcore_model` 做了 assert 拦截。

- In reproduction scenarios, **do not** enable flags that discard optimizer / RNG state — exact resume needs the full optim and rng (see [Resume Training](resume.md)).
- Some checkpoint options introduce precision loss that breaks post-resume determinism; gcore already asserts against `skip_save_mcore_model` in deterministic mode.

## 快速开始 / Quick start

```bash
cd /work/wepsdl/gcore-xxx
bash tests/test_alignment_v4/rl/gcore/scripts/run.sh
```

脚本核心就是在常规 GRPO 入口上追加确定性开关：

The script simply appends the deterministic switch to the regular GRPO entry:

```bash
python3 -u gpatch_v4/entry/train_lm_grpo.py \
    --config-path="tests/test_alignment_v4/rl/gcore/config" \
    --config-name="align" \
    training.apply_deterministic_mode=True \
    training.attention_backend=flash \
    "$@"
```
