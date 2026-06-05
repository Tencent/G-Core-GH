# core — 公共算法与基础设施

跨模块共享的核心逻辑：模型架构常量、分布式并行状态、advantage 计算、DP 负载均衡等。

## 文件说明

| 文件 | 说明 |
|------|------|
| `constants.py` | `MODEL_ARCH` 枚举 — 所有支持的模型架构标识（llama、qwen 系列、flux、oteam、bagel 等），命名规范跟随 HuggingFace Transformers |
| `parallel_state.py` | 分布式并行状态管理：`cpu_barrier`、`is_mp_and_cp_head`、process group 工具函数 |
| `device/` | 设备管理抽象与后端实现（子目录） |
| `device/__init__.py` | 设备后端入口：自动发现 `*_priv` 后端，暴露 `get_device_module()`、`is_cuda()`、`get_dist_backend()` 等 |
| `device/protocol.py` | `CudaDeviceBackend` — 默认 CUDA 后端协议 |
| `device/*_priv.py` | 闭源设备后端（打包时剥离） |
| `advantage_helper.py` | `AdvantageContext` / `AdvantageResult` 数据结构，`get_advantage_fn` 返回对应的 advantage 计算函数 |
| `advantage_impl.py` | 各种 advantage 实现（GRPO、PPO、GDPO 等） |
| `dp_balancing.py` | Data Parallel 负载均衡 |
| `seqlen_balancing.py` | 序列长度均衡 |
| `smart_pad_helper.py` | 智能 padding |
| `mappings.py` | 通用映射工具 |
| `correction_helper.py` | 修正辅助函数 |

## MODEL_ARCH 命名规范

- 从 HuggingFace Transformers 获取 `model_type`，避免不规范命名（如 `qwen2p5vl`）
- 内部模型用 `oteam4_3`、`oteam4_4` 等命名
- 类名示例：`model_arch='oteam4_5_moe'` → `class Oteam4_5_MoeXxxYyy`
