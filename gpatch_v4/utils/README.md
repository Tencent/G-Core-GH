# utils — 工具函数库

跨模块共享的工具函数集合。

## 文件说明

| 文件 | 说明 |
|------|------|
| `common_utils.py` | 通用工具：日志（`log`、`logging_rank0`）、内存管理（`clear_memory`、`logging_memory_usage`）、性能计时（`sync_cuda_and_get_time`、`perf_time`、`profile_memory_and_time`）、动态导入（`import_fn_from_path`、`import_mod_from_path`） |
| `communication_utils.py` | 分布式通信工具：`BroadcastUtils`（rollout batch 广播）、`allreduce_loss_across_data_parallel_group`、`average_losses_across_data_parallel_group` |
| `training_utils.py` | 训练工具：`check_rollout_batches`、`masked_mean_list`、`masked_global_statistics_list`、`cpu_dict` 等 |
| `ppo_utils.py` | PPO 算法工具：`calculate_kl_penalty`、`create_response_mask` |
| `report_utils.py` | 实验上报：`TrainReporterSingleton`（WandB 等）、`init_train_reporter_singleton` |
| `timer_utils.py` | 计时器：`TimerSingleton`、`init_timer_singleton`、`record_time_to_metrics` |
| `filter_samplings.py` | 采样过滤策略：`best_and_worst`、`truncated_test` |
| `resumable_distributed_sampler.py` | 可恢复的分布式采样器（支持 checkpoint 恢复后从断点继续） |
| `reward_redist.py` | 奖励重分配工具 |
| `reloadable_process_group.py` | 可重建的 process group：`reload_process_groups`、`destroy_process_groups`、`monkey_patch_torch_dist` |
| `test_utils.py` | 测试 / 调试工具：`save_data` |
| `str_utils.py` | 字符串工具：`contains_renderable_field` 等模板字段检查 |
| `data_manipulate_utils.py` | 数据操作工具：`union_two_dict`、TensorDict 操作等 |
| `constants.py` | 工具常量：`GenerateStopReason` 枚举等 |
| `packages.py` | 包版本检测：`_is_package_available`、`_get_package_version` |
| `flops_counter.py` | FLOPs 计算器 |
| `flops_counter_priv.py` | FLOPs 计算器（闭源部分） |
