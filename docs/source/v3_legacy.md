# Trainer V3 (Legacy)

```{note}
V3 使用 mpirun 多进程编排，已被 V4 的 Ray-based 架构取代。
新项目请优先使用 [V4](math_grpo_v4_demo.md)。
V3 仍可正常使用，但不再新增功能。

V3 uses mpirun multi-process orchestration, which has been superseded by V4's Ray-based architecture.
New projects should prefer [V4](math_grpo_v4_demo.md).
V3 remains functional but receives no new features.
```

## V3 Overview / V3 概览

V3 将 GRPO 训练拆分为三个独立进程：Actor、Sampler、Critic/RM，各自通过 `mpirun` 在不同节点上启动，使用 `tools/auto_place.py` 生成 GPU 布局配置。

V3 splits GRPO training into three independent processes: Actor, Sampler, and Critic/RM, each launched via `mpirun` on different nodes, with GPU placement configured by `tools/auto_place.py`.

```
┌──────────────────────────────────────────────────────────────┐
│                   V3 Architecture Overview                    │
│                                                              │
│   mpirun ──┬── Actor (Megatron-LM, train_ppo_actor.py)      │
│            ├── Sampler (vLLM/SGLang, train_ppo_sampler.py)   │
│            └── Critic/RM (Megatron-LM, train_ppo_critic.py)  │
│                                                              │
│   配置: tools/auto_place.py → place-config/                   │
│   启动: 分别 mpirun 三个角色                                    │
└──────────────────────────────────────────────────────────────┘
```

## V3 Examples / V3 例子

```{toctree}
:maxdepth: 1

math_sft.md
math_grpo.md
math_grpo_gen_rm.md
vqa.md
```

## Contents / 内容

```{toctree}
:maxdepth: 1

trainer.md
trainer-internal.md
trainer_v3_args.md
gdataset.md
datasets_refactor.md
resume.md
sampling_strategy.md
rollout_router_replay.md
dump_metrics.md
tpo.md
```

## V3 API Reference

```{toctree}
:maxdepth: 1

api/trainer.rst
api/gdataset.rst
```
