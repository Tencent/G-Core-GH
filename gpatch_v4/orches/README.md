# orches — Ray 编排层

Ray 集群初始化、placement group 管理、Actor group 创建与生命周期管理。是 Trainer 与 Actor 之间的桥梁。

## 核心流程

```
orches.init(config)                       # 初始化 Ray runtime
PlacementGroupManager.create(...)   # 创建 placement group
TrainGroup / SamplerGroup / ...     # 在 PG 中部署 Actor
orches.shutdown()                   # 关闭 Ray
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `__init__.py` | `init()`、`shutdown()`、`get()`、`get_actor()`、`get_queue()` — Ray 运行时薄封装 |
| `base_actor.py` | `BaseActor` — 所有 Ray Actor 的基类，提供分布式初始化、checkpoint 等 |
| `train_actor.py` | 训练 Actor 的 Ray remote 装饰逻辑 |
| `placement_group.py` | Placement group 管理：colocate / disaggregated 策略 |
| `resource_allocator.py` | `ResourceAllocation` — 每个角色的 GPU 资源意图（节点数、每节点 GPU 数、每副本最小 GPU 数）；`allocation_from_config` 从顶层 config 导出 |
| `train_group.py` | 训练 Actor 组：根据任务类型选择 Actor 类并批量创建 |
| `sampler_group.py` | 采样 Actor 组 |
| `gen_rm_group.py` | Generative RM Actor 组 |
| `bt_rm_group.py` | Batch RM Actor 组 |
| `teacher_group.py` | 蒸馏教师 Actor 组 |
| `kv_store_group.py` | KV Store Actor 组 |
| `training_plt_group.py` | 训练平台上报 Actor 组 |
| `custom_actor_registry.py` | 用户自定义 Actor 注册：从配置路径动态加载 Actor 类 |
| `data_source.py` | `DataSourceBase` — rollout 数据源抽象基类，支持 partial rollout 缓冲 |
| `failure.py` | `FailureType` / `FailureEvent` — 分布式训练故障事件定义 |
| `node_replacer.py` | `NodeReplacer` — 节点驱逐与补充的抽象接口，用于故障自动恢复 |
| `exceptions.py` | 编排异常类型：re-export `RayActorError` / `RayTaskError`，避免外部直接依赖 Ray |
| `utils.py` | 编排辅助工具 |
