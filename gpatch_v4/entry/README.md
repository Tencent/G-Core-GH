# entry — CLI 入口

Hydra `@hydra.main` 入口脚本，每种训练 / 评估 / 推理任务对应一个入口文件。通常由启动脚本直接调用。

## 文件说明

| 文件 | 对应 Config | 对应 Trainer | 说明 |
|------|------------|-------------|------|
| `train_lm_grpo.py` | `RlConfig` | `GrpoTrainer` | LM GRPO 训练 |
| `train_lm_grpo_single_ctrl.py` | `RlConfig` | `GrpoSingleCtrlTrainer` | LM GRPO 单控制器训练（colocate / disaggregated） |
| `train_vlm_grpo.py` | `RlConfig` | `GrpoTrainer` | VLM GRPO 训练 |
| `train_lm_finetune.py` | `FinetuneConfig` | `FinetuneTrainer` | LM SFT |
| `train_vlm_finetune.py` | `FinetuneConfig` | `FinetuneTrainer` | VLM SFT |
| `train_dpo.py` | `DpoConfig` | `DpoTrainer` | DPO 训练 |
| `train_on_policy_distill.py` | `OnPolicyDistillConfig` | `OnPolicyDistillTrainer` | On-policy 蒸馏 |
| `train_off_policy_distill.py` | `OffPolicyDistillConfig` | `OffPolicyDistillTrainer` | Off-policy 蒸馏 |
| `train_t2i_grpo.py` | `T2iRlConfig` | `T2iGrpoTrainer` | T2I GRPO |
| `train_t2i_dpo.py` | `T2iDpoConfig` | `T2iDpoTrainer` | T2I DPO |
| `train_t2i_edit_sft.py` | `T2iEditSftConfig` | `T2iEditSftTrainer` | T2I 编辑 SFT |
| `train_bagel.py` | `BagelConfig` | `BagelTrainer` | BAGEL 训练 |
| `train_bagel_t2i_grpo.py` | `BagelT2iRlConfig` | `T2iGrpoTrainer` | BAGEL T2I GRPO |
| `eval_entry.py` | `EvaluateConfig` | `EvaluateRunner` | 模型评估 |
| `infer_entry.py` | `InferenceConfig` | `InferenceRunner` | 纯推理 |
| `verify_dataloader.py` | — | — | 数据加载验证工具 |

## 典型调用

```bash
python -m gpatch_v4.entry.train_lm_grpo \
    --config-path /path/to/yaml \
    --config-name my_config \
    policy.hf_model_path=/path/to/model
```
