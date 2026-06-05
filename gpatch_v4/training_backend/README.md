# training_backend — 训练引擎

封装训练的核心执行逻辑：模型初始化、前向 / 反向传播、优化器步进、checkpoint 管理。`TrainingEngineFactory` 根据 `config.training.training_backend` 选择后端。

## 支持的后端

| training_backend | 引擎类 | 适用场景 |
|-----------------|--------|---------|
| `fsdp2` | `Fsdp2EngineLm` | 文本 LM/VLM（FSDP2 分布式） |
| `fsdp2` | `Fsdp2EngineT2i` | T2I 模型（FSDP2 分布式） |
| `mcore` | `McoreEngine` | 文本 LM/VLM（Megatron-Core TP/PP/CP） |

## 目录结构

```
training_backend/
├── __init__.py                    # TrainingEngineFactory
├── base_engine.py                 # 引擎基类
├── loss_factory.py                # Loss 函数工厂（LOSS_FUNC_REGISTRY）
├── vocab_parallel_entropy.py      # 词表并行 entropy 计算
├── common/
│   ├── swap_mixin.py              # 模型/优化器 CPU offload mixin
│   ├── omni_training_utils.py     # Omni 模型训练工具
│   └── omni_training_utils_priv.py # Omni 训练工具（闭源）
├── fsdp2_backend/
│   ├── __init__.py                # 导出 Fsdp2EngineLm / Fsdp2EngineT2i
│   ├── fsdp2_engine_lm.py        # FSDP2 LM 引擎
│   ├── fsdp2_engine_t2i.py       # FSDP2 T2I 引擎
│   ├── fsdp2_engine_vlm.py       # FSDP2 VLM 引擎
│   ├── checkpoint.py             # FSDP2 checkpoint
│   ├── checkpoint_t2i.py         # FSDP2 T2I checkpoint
│   ├── optimizer.py              # FSDP2 优化器
│   ├── lr_scheduler.py           # FSDP2 LR scheduler
│   ├── swap.py                   # FSDP2 CPU swap 入口
│   ├── fsdp2_swap_impl.py        # FSDP2 CPU swap 实现
│   ├── ema_model_t2i.py          # FSDP2 EMA（T2I）
│   ├── engine_helper_t2i.py      # T2I 引擎辅助
│   ├── fsdp2_utils_bagel.py      # BAGEL FSDP2 工具
│   └── mixin.py                  # FSDP2 公共 mixin
└── megatron_backend/
    ├── __init__.py                # 导出 McoreEngine
    ├── mcore_engine.py            # Megatron-Core 引擎
    ├── megatron_bridge.py         # Megatron 与 HF 模型桥接
    ├── checkpoint.py              # Megatron checkpoint
    ├── model_forward.py           # Megatron forward 逻辑
    ├── optimizer.py               # Megatron 优化器
    ├── mcore_peft.py              # PEFT 支持
    ├── router_replay_manager.py   # MoE 路由重放
    ├── mbridge.py                 # mbridge 集成
    ├── mcore_swap_impl.py         # Megatron CPU swap 实现
    ├── megatron_utils.py          # Megatron 工具函数
    └── mixin.py                   # Megatron 公共 mixin
```
