# 多机多卡仅推理模式

本任务用于**纯推理模式**，即只进行模型推理采样，不进行训练。适用于：
- 批量采样
- 模型评测

## 快速启动

### 1. 初始化 Ray 集群

在 `gcore-dev` 根目录执行：

初始化 ray 集群：
```bash
bash tasks/infer_only/init.sh
```

> **注意**：如果启动过程中出现卡顿，可以先手动初始化 Ray：
> ```bash
> python3
> >>> import ray
> >>> ray.init()
> >>> exit()
> ```

再跑上述初始化命令

### 2. 启动推理任务

```bash
bash tasks/infer_only/infer_only.sh <config_name>
```

例如启动 DeepSeek-R1 推理任务：

```bash
bash tasks/infer_only/infer_only.sh infer_only_dpsk
```

`<config_name>` 对应 `tasks/infer_only/` 目录下的 yaml 配置文件名（不含 `.yaml` 后缀）。

可用参考配置：
- `infer_only_dpsk` - DeepSeek-R1
- `infer_only_gptoss` - GPT-OSS-120B
- `infer_only_qw3next` - Qwen3-Next-80B
- `infer_only_qwq` - QwQ-32B

## YAML 配置说明

### data - 数据配置

```yaml
data:
  data_pathes:
    - "hf-hub/openai/gsm8k-jsonl/eval"  # 数据集路径
  py_path: "tasks/infer_only/simple_dataset.py"  # 数据加载脚本
  fn_name: "get_dataset_and_dataloader"  # 数据加载函数名
```

### data loader 数据格式
jsonl 文件，格式可以自定义

默认格式可以参考 tasks/infer_only/simple_dataset.py:collate_batch_data
```
    Input: [{"query": "...", "search_res": "..."},
            {"query": "...", "search_res": "..."},
            {"query": "...", "search_res": "..."},
            ...
           ]
```
prompt 拼接规则可自定义

默认格式参考 tasks/infer_only/sample_demo.py:generate_func
```
prompt = f"问题：{query}\n\n参考资料：\n{search_res}\n\n请基于上述参考资料回答问题："
```

### sampler - 推理引擎配置

```yaml
sampler:
  backend: sglang  # 推理后端，仅支持 sglang
  sampler_type: "sampler"
  
  # 分布式配置
  dist_config:
    nnodes: 4  # 节点数
    tensor_model_parallel_size: 16  # TP 并行度
    num_gpus_per_node: 8  # 每节点 GPU 数
  
  # 模型信息
  model_info:
    - model_arch: "deepseek"  # 模型架构
      hf_model_path: hf-hub/deepseek-ai/DeepSeek-R1  # 模型路径
      gen_rollout_py_path: "tasks/infer_only/sample_demo.py"  # 采样脚本
      gen_rollout_fn_name: "generate_func"  # 采样函数名

  # 推理引擎参数
  infer_engine_configs:
    - dist_config: # 对齐 sampler 侧的配置
        nnodes: 4
        tensor_model_parallel_size: 16
        num_gpus_per_node: 8
      gpu_memory_utilization: 0.8  # GPU 显存利用率
      max_running_requests: 192  # 最大并发请求数
      temperature: 0.6  # 采样温度
      top_p: 0.95
      generate_max_tokens: 7168  # 最大生成 token 数
      use_fast_tokenizer: False
```

### infer_result - 输出配置

```yaml
infer_result:
  output_dir: "infer_only_results/dpsk"  # 输出目录
  enable_think_mode: False  # 是否启用思考模式
  sampling_repeat: 16  # 每个样本重复采样次数
```
