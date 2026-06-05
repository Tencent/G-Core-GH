# evaluate — 评估与推理 Runner

提供评估和推理的高层运行器，负责 Ray placement group 创建、Actor 编排和结果收集。

## 文件说明

| 文件 | 说明 |
|------|------|
| `evaluate_runner.py` | `EvaluateRunner` — 离线评估：创建 sampler，运行多次采样，收集 reward 并输出结果 |
| `inference_runner.py` | `InferenceRunner` — 纯推理：加载数据、调用 sampler 生成、保存输出 |

## 与 Trainer 的关系

- `EvaluateRunner` 可独立使用，也可在 RL 训练中被调用（`trainer.run_eval`）
- `InferenceRunner` 适用于独立推理场景，不依赖训练循环
