# generation_backend — 推理引擎

统一的推理引擎抽象层，封装 vLLM 和 SGLang 两种后端，对上层提供一致的接口。

## 核心类

- `InferEngine` — 统一推理封装。构造参数为 `(infer_engine, model_path, infer_engine_role, placement_type)`。提供 `async_generate`、`wait_and_get_async_generate_output`、`update_weights`、`sleep` / `wake_up` 等方法。由 `from_engine_args(...)` 工厂方法创建。

## 文件说明

| 文件 | 说明 |
|------|------|
| `infer_engine.py` | `InferEngine` 统一封装，处理 SGLang import patch、信号 hack 等兼容性问题 |
| `vllm_engine.py` | vLLM 后端适配 |
| `sglang_engine.py` | SGLang 后端适配 |
| `routed_experts_utils.py` | MoE 路由专家辅助工具 |

## 使用方式

```python
engine = InferEngine.from_engine_args(infer_engine_config, ...)
outputs = await engine.async_generate(requests)
await engine.update_weights(state_dict)
```
