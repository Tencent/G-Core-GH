# Gcore ReTool

gcore 对 [ReTool](https://arxiv.org/abs/2504.11536) multi-turn tool calling agent RL 训练的实现。参考 [verl retool](https://github.com/verl-project/verl-recipe/tree/main/retool)

## 数据集
- train: [dapo_math_17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
- eval: [aime24](https://huggingface.co/datasets/math-ai/aime24)

使用 `tasks/math_rl_v3/qwen/preprocess_sft_data_phase_1.py` 将原数据集转为 jsonl 文件，设置 `--gdatasetv4-train-metadata-file`, `--gdatasetv4-eval-metadata-file` 数据集路径。

## External / Internal Agent

- Internal Agent: rollout agent 自行管理多轮对话状态以及tool parse，采用 async_generate 离线api与 gcore 推理引擎交互

- External Agent: rollout agent 采用 openai chat/completions api 与 gcore 推理引擎交互，多轮对话状态管理以及tool parse在gcore推理引擎内完成

```shell
# external experiment
bash tasks/retool/external_agent/mpirun_retool.sh
# internal experiment
bash tasks/retool/internal_agent/mpirun-init-ray.sh
bash tasks/retool/internal_agent/mpirun_retool.sh
```
## NOTE

当前 tasks/retool/local_sandbox_server.py 并未运行在单独的 sandbox 容器中，可以执行模型输出的代码，有一定安全风险，**不要放在生产环境使用**。可以参考 https://github.com/bytedance/SandboxFusion 构建安全的沙箱环境。
 