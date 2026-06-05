# rollout — Rollout 管理

管理 RL 训练中的 rollout 生成过程，包括 DP 聚合、partial rollout、请求分组和早停协调。

## 核心类

| 类 | 说明 |
|----|------|
| `RolloutManager` | 管理 rollout 生成的完整流程：DP 负载均衡、partial rollout 拼接、请求分发与结果收集 |
| `RolloutCoordinator` | 协调多路 rollout 收集，当达到目标数量时 abort 采样引擎以节省计算 |
| `RolloutRequestGroup` | 将请求按组管理，支持分批发送 |

## 文件说明

| 文件 | 说明 |
|------|------|
| `rollout_manager.py` | `RolloutManager` — DP allgather、partial rollout、负载均衡 |
| `rollout_coordinator.py` | `RolloutCoordinator` — 异步锁保护的 rollout 计数与早停 |
| `request_group.py` | `RolloutRequestGroup` — 请求分组数据结构 |
