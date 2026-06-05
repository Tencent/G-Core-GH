# trainer — 高层训练器

训练外循环的实现：初始化 Ray 集群、创建 Actor、执行训练步（rollout → reward → advantage → policy update）、checkpoint 和评估。

## 已有 Trainer

| Trainer 类 | 说明 |
|------------|------|
| `GrpoTrainer` | 文本 GRPO RL 训练（LM / VLM） |
| `GrpoSingleCtrlTrainer` | 文本 GRPO 单控制器训练（colocate / disaggregated，支持 async rollout） |
| `FinetuneTrainer` | SFT 微调 |
| `DpoTrainer` | DPO 训练 |
| `OnPolicyDistillTrainer` | On-policy 蒸馏 |
| `OffPolicyDistillTrainer` | Off-policy 蒸馏 |
| `T2iGrpoTrainer` | T2I GRPO 训练 |
| `T2iDpoTrainer` | T2I DPO 训练 |
| `T2iEditSftTrainer` | T2I 编辑 SFT |
| `BagelTrainer` | BAGEL 训练 |
| `OmniBaseTrainer` | Omni 模型训练基类 |

## 文件说明

| 文件 | 说明 |
|------|------|
| `base_trainer.py` | `BaseTrainer` — 公共训练流程：init → data → train loop → checkpoint → eval |
| `trainer_mixin.py` | 训练器公共 mixin |
| `helper.py` | Hydra launch 辅助、多节点启动工具 |
| `grpo_trainer.py` | GRPO 训练循环：rollout → advantage → PPO update |
| `grpo_single_ctrl_trainer.py` | GRPO 单控制器训练循环 |
| `finetune_trainer.py` | SFT 训练循环 |
| `dpo_trainer.py` | DPO 训练循环 |
| `on_policy_distill_trainer.py` | On-policy 蒸馏训练 |
| `off_policy_distill_trainer.py` | Off-policy 蒸馏训练 |
| `t2i_grpo_trainer.py` | T2I GRPO 训练 |
| `t2i_dpo_trainer.py` | T2I DPO 训练 |
| `t2i_edit_sft_trainer.py` | T2I 编辑 SFT |
| `bagel_trainer.py` | BAGEL 训练 |
| `omni_base_trainer.py` | Omni 模型训练基类 |

## 典型 GRPO 训练循环

```
1. orches.init(config)                    # 初始化 Ray
2. 创建 placement group + Actor groups
3. for step in train_steps:
   a. rollout_generator.generate()  # 采样 + RM 打分
   b. generate_ppo_data()           # 计算 advantage / returns
   c. policy_engine.train_step()    # 策略梯度更新
   d. update_weights()              # 同步权重到 sampler
   e. (可选) run_eval()             # 评估
   f. save_checkpoint()             # 保存 checkpoint
4. orches.shutdown()
```
