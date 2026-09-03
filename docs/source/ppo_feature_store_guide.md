# PpoFeatureStore 功能说明

> 包路径：`gpatch_v4/core/ppo_feature_store/`  
> 承接 plan：`train_extra_state_generic_plan.md`（由 EPO 专用泛化为任意标量 feature）  
> 日期：2026-08-14；**修订 2026-08-17**（`feature_store_enable` 总开关 + EPO knobs 迁出 `ppo_config`）

---

## 修订记录（2026-08-17）

相对首版 guide，实现侧已变、本文已对齐的要点：

1. **`ppo.feature_store_enable`（默认 `False`）** 为总开关：关则不 init 单例、interval 空操作、不 save/load、不上报 `extra/*`。
2. **`set_ppo_feature_store_enabled` / `is_ppo_feature_store_enabled`**：`get_ppo_feature_store()` 在未 enable 时 **raise**（不再静默建单例）。
3. **Save 门控**：仅当 `has_persisted_features()` 为真才写 sidecar（只有 `set_step_local`、无 history/自由 set 时不落盘）。
4. **EPO 配置不在 `PpoConfig`**：`epo_*` 读 `config.task`（yaml `task:`；Hydra 下常为 **dict**）；core 无 `epo_enable`。
5. **Metrics**：`ReportMetricsMixin.report_extra_metrics`（原 `_record_extra_metrics`）；`extra/{name}` = **本步 finalize 值**，不是 history mean。

---

## 1. 整体：解决什么问题

GRPO / PPO 训练里，自定义 **advantage** 与 **loss** 经常需要跨 microbatch、跨 DP rank、跨 ppo_step 维护一些**附属标量状态**，例如：

- 历史熵均值（EPO baseline）
- advantage 的 mean / min / max 统计
- 其它业务自定义标量

这些量**不是**模型权重，也不适合塞进主 checkpoint 的 optimizer state。`PpoFeatureStore` 提供：

1. **进程内全局 KV**：任意名字的标量 feature
2. **ppo_step 区间语义**：一步内多次 `record` → 步末统一 DP reduce → 写入 `history`
3. **可注册 reduce**：内置 `mean|sum|max|min`，业务可 `@register_feature_reduce`
4. **独立附属 ckpt**：与 iter 主 checkpoint 同目录另存 `ppo_feature_store.pt`，resume 可恢复 history

**History feature 只存标量**（`float`）。张量需调用方先收成标量再 `record`。  
**只支持** `ppo_step` **轴**（不支持 `train_step` 轴）。

另有 **步内 ephemeral tensor**：`set_step_local` / `get_step_local`。无 pending、无 reduce、无 history、不进 ckpt；interval enter/exit 清掉。Store 不做 DP/PP 通信。

---

## 1.5 总开关 `feature_store_enable`

| 项 | 行为 |
| --- | --- |
| 配置 | `ppo.feature_store_enable: bool = False`（`gpatch_v4/configs/ppo_config.py`） |
| Actor init | `set_ppo_feature_store_enabled(flag)`；`True` → `self.feature_store = get_ppo_feature_store()`，否则 `None` |
| Interval | `ppo_step_interval(..., enabled=self.config.ppo.feature_store_enable)` |
| Load / save / metrics | 均先查 `feature_store_enable`；save 再查 `has_persisted_features()` |
| Task 侧 | custom loss / advantage 若调用 `get_ppo_feature_store()`，**必须**把开关打开，否则 `RuntimeError` |

YAML 示例（e2e）：

```yaml
ppo:
  feature_store_enable: true
```

---

## 2. 模块结构

```text
gpatch_v4/core/ppo_feature_store/
├── __init__.py     # 对外 re-export
├── keys.py         # key 命名、ckpt 文件名、version
├── reduce.py       # reduce 注册表 + 内置 mean/sum/max/min
├── interval.py     # PpoStepInterval 上下文管理器
└── state.py        # PpoFeatureStore 主体 + enable 门控 + finalize 同步
```


| 模块         | 职责                                                                           |
| ---------- | ---------------------------------------------------------------------------- |
| `keys`     | `{feature}.pending/.history/.reduce/.history_axis` 约定；`ppo_feature_store.pt` |
| `reduce`   | `register_feature_reduce` / `get_feature_reduce`；默认 DP group                 |
| `interval` | enter 清 pending；exit 全量 finalize + 广播                                        |
| `state`    | 单例 store + enable flag；record / history / ckpt / broadcast                    |


---



## 3. 生命周期（从整体到一步内）

```text
                    ┌─────────────────────────────────────────┐
                    │  with ppo_step_interval(ppo_step=t,     │
                    │       enabled=feature_store_enable):    │
                    │                                         │
 enter ───────────► │  begin: 记下 ppo_step，清空 *.pending + _step_local │
                    │                                         │
                    │  advantage / loss / 任意调用方:           │
                    │    store.configure(name, reduce=...)    │
                    │    store.record(name, value, weight=…)  │
                    │    baseline = store.get_history_mean()  │
                    │    store.set_step_local(name, tensor)   │
                    │    t = store.get_step_local(name)       │
                    │                                         │
 exit (无异常) ────► │  finalize_all_pending + PP 同步广播       │
                    │  每个非空 pending → DP reduce → append   │
                    │  到 {name}.history；写入 iv.final_values │
                    └─────────────────────────────────────────┘
                              │
                              ▼
              actor.report_extra_metrics → extra/{name}, extra/{name}_history_len
              周期性 save: iter_XXXXXXX/ppo_feature_store.pt（有 persist 内容时）
```

Actor 接线（sync）：`GrpoTrainActor.train_one_ppo_step` 用 `ppo_step_interval` **包住** rollout（含 advantage）+ policy train；exit 后再 `report_extra_metrics` + `log_and_report`。Async actor 同理。

这样 advantage 里 `record` 的 pending **不会**被 train 阶段的 begin 清掉。

---



## 4. 数据模型（Key 布局）

每个 feature 名 `F` 对应一组派生 key：


| Key              | 形态                      | 是否落盘 | 含义                         |
| ---------------- | ----------------------- | ---- | -------------------------- |
| `F.pending`      | `list[(value, weight)]` | 否    | 本 ppo_step 本地贡献            |
| `F.history`      | `list[float]`           | 是    | 每步 finalize 后一条            |
| `F.reduce`       | `str`                   | 是    | reduce 名字（须能在 registry 解析） |
| `F.history_axis` | `"ppo_step"`            | 是    | 目前仅允许该值                    |
| `ppo_step`       | `int \| None`           | 是    | 当前 / 最近 interval 的 step     |


另外可用自由 `get` / `set`（例如 EPO 的 `epo.axis_total`），只要不是 `*.pending` 就会 persist。

`set_step_local` 走独立的 `_step_local` dict，**不在**上表、**不进** `state_dict`。

**`has_persisted_features()`**：除 `ppo_step` / `train_step` / `*.pending` 外是否还有可落盘 key。Save 用此判断，避免空 sidecar。

**pending 语义（与 reduce 绑定）**：

- `mean`：`(sum_contribution, count)`；finalize 时 Σsum / Σcount（跨 DP）
- `sum` / `max` / `min`：`(value, 1.0)`；weight 在 `record` 时忽略

**history 读法**：`get_history_mean(F)` = `mean(history)`，与该步用的 reduce **无关**——是跨 ppo_step 的简单平均。

---



## 5. 接口设计



### 5.1 单例入口

```python
from gpatch_v4.core.ppo_feature_store import (
    get_ppo_feature_store,
    set_ppo_feature_store_enabled,
    is_ppo_feature_store_enabled,
    reset_ppo_feature_store_for_test,  # 单测用：清单例并 enable=True
)

set_ppo_feature_store_enabled(True)   # 生产由 actor 按 config 调用
store = get_ppo_feature_store()       # enable=False → RuntimeError
```



### 5.2 配置与写入

```python
store.configure("entropy", reduce="mean")   # 显式绑定；同名冲突 assert
store.record("entropy", value_sum, weight=token_count)
# 或首次 record 时带 reduce=（等价 configure）
store.record("latency", 0.12, reduce="max")
```


| 方法                                                  | 作用                                           |
| --------------------------------------------------- | -------------------------------------------- |
| `configure(feature, reduce=...)`                    | 绑定 reduce；已绑定且不一致 → `AssertionError`         |
| `record(feature, value, weight=1.0, reduce=None)`   | 追加 pending；未 configure 默认 `mean`             |
| `record_weighted_sum(feature, value_sum, count)`    | `record(..., reduce="mean")` 别名              |
| `get_history_mean(feature)`                         | 历史简单均值；无 history → `None`                    |
| `has_persisted_features()`                          | 是否有除 step 键 / pending 外的可落盘内容                 |
| `get` / `set` / `append` / `has` / `keys` / `clear` | 底层 KV（会 persist；不要用来存步内 tensor）              |
| `set_step_local(name, tensor)`                      | 本 ppo_step 写一次；`detach().clone()`；二次写 assert |
| `get_step_local(name)`                              | 读步内 tensor；未 set → `KeyError`                |
| `clear_step_local()`                                | 清空 `_step_local`；interval enter/exit 会调      |




### 5.3 Interval（推荐唯一用法）

```python
from gpatch_v4.core.ppo_feature_store import ppo_step_interval

with ppo_step_interval(ppo_step=t, enabled=True) as iv:
    ...  # record / get_history_mean / set_step_local / get_step_local
# 正常退出后：
#   iv.final_values[feature] = float | None
#   _step_local 已清空（异常退出也会清）
```

- `enabled=False`：空操作（不 begin / 不 finalize）；actor 用 `feature_store_enable` 传入
- 异常退出：不 finalize（pending 丢弃语义由下次 enter 清空保证）；`_step_local` 仍会清掉

底层也有 `begin_ppo_step_interval` / `finalize_feature_interval` / `finalize_all_pending_features`，业务侧优先用 context manager。

### 5.4 Finalize + 分布式同步

`finalize_all_pending_features_and_sync()`：

1. **仅 pipeline last stage** 做 DP reduce + append history
2. 经 `cpu_group` 从 last rank **broadcast** 整份 `state_dict` + `values`
3. 其它 PP stage `load_state_dict`，保证全员 history 一致

单步多 feature：会先 `all_gather_object` 各 rank 的 pending 名并集，避免「某 rank 没 record 就不参与 collective」的分叉。

### 5.5 Reduce 注册

```python
from gpatch_v4.core.ppo_feature_store import register_feature_reduce

@register_feature_reduce("my_p99")
def reduce_p99(local_values, local_weights, group):
    # 签名: (list[float], list[float]|None, ProcessGroup|None) -> float|None
    ...
    return float(...)

# 覆盖内置须显式 override=True
register_feature_reduce("mean", my_mean, override=True)
```


| 内置     | 本地               | DP      |
| ------ | ---------------- | ------- |
| `mean` | Σvalue / Σweight | SUM/SUM |
| `sum`  | Σvalue           | SUM     |
| `max`  | max              | MAX     |
| `min`  | min              | MIN     |


自定义 reduce：**所有相关 DP rank 都必须调用**（内部若有集体通信则全员进）。

### 5.6 Checkpoint

```python
# 写（仅 rank0 落盘；actor 侧另有 has_persisted_features 门控）
store.save_to_ckpt(iter_dir, ppo_step)

# 读：优先 ppo_feature_store.pt，兼容旧 train_extra_state.pt
ok = store.load_from_ckpt(iter_dir, ppo_step, strict_step=True)
store.broadcast()  # resume 后向其它 rank 同步
```

Persist 内容：`version`、`ppo_step`、非 pending 的 `data`（含 `.history` / `.reduce` / 自由 set）。  
`strict_step=True` 时文件内 `ppo_step` 必须与期望一致。

Actor 侧：`_maybe_load_ppo_feature_store` / `_maybe_save_ppo_feature_store`；metrics 经 `report_extra_metrics`，前缀 `extra/{name}`。

---



## 6. 如何使用（调用方视角）



### 6.1 最小示例（自定义 loss / advantage）

```python
from gpatch_v4.core.ppo_feature_store import get_ppo_feature_store

FEATURE = "my_stat"

def my_hook(...):
    store = get_ppo_feature_store()
    # 读历史基线（本步尚未 finalize，不含当前 pending）
    baseline = store.get_history_mean(FEATURE)

    # 写入本步贡献（tensor → 标量）
    store.record(FEATURE, local_sum, weight=local_count, reduce="mean")
    ...
```

Interval 由 **actor** 包好；task 侧只需 `record` / `get_history_mean` / `set_step_local` / `get_step_local`，一般**不必**自己套 `ppo_step_interval`。

前提：yaml / config 已设 `ppo.feature_store_enable: true`。

步内 tensor（advantage 写、loss 读；已 DP-reduce；不进 ckpt）::

```
store.set_step_local("sample_w", already_dp_reduced)  # 一步一次
w = store.get_step_local("sample_w")                  # 同一步内可多次
```



### 6.2 已有参考实现

**EPO custom loss**（`tasks/math_rl_v4/epo_grpo_loss.py`）：

- feature 名字符串 `"epo"`（仅 task 侧约定，非 core 常量）
- knobs 从 **`config.task`** 读（`_task_epo_settings`；支持 dict / namespace）：  
  `epo_mask_mode` / `epo_min_ratio` / `epo_max_ratio` / `epo_out_range_penalty` /  
  `epo_entropy_smooth_coeff` / `epo_enable_smooth_weights`
- `record("epo", entropy_sum, weight=token_count, reduce="mean")`
- `baseline_h = get_history_mean("epo")` 做熵带 penalty
- phase-weight 分母：优先 `epo.axis_total`，否则回退 `training.total_ppo_step`

**Advantage 统计 demo**（`tasks/math_rl_v4/custom_advantage.py` → `custom_grpo_advantage_with_adv_stats`）：

```python
store.record("adv_mean", vals.sum(), weight=vals.numel(), reduce="mean")
store.record("adv_min", vals.min(), reduce="min")
store.record("adv_max", vals.max(), reduce="max")
```



### 6.3 上报字段

Interval exit 后 actor 写入（示意）：

```text
extra/{feature}              # 本步 finalize 标量（非 history mean）
extra/{feature}_history_len  # finalize 后 history 长度
```



### 6.4 单测 / e2e

- `tests/test_gpatch_v4/test_ppo_feature_store.py` — 基础 / reduce / ckpt / interval / enable / `set_step_local`
- `test_custom_advantage_adv_stats.py`、`test_epo_loss.py` — 业务单测
- `test_epo_grpo_e2e.py` / `test_epo_grpo_async_e2e.py` — sync/async history + sidecar（需 `feature_store_enable: true` + `task.epo_*`）

测试前后调用 `reset_ppo_feature_store_for_test()` 清单例并 enable。

---



## 7. 设计约束与常见坑

1. History feature **只存标量**；张量先本地 fold。步内 tensor 用 `set_step_local`，不要 `set`/`record`。
2. **同 feature 的 reduce 不可改**（configure / record 二次不一致会 assert）。
3. `get_history_mean` **不含当前 pending**——本步刚 `record` 的值要等 exit finalize 后才进 history。
4. **不要在** `generate_ppo_data` **内再套一层 interval**（嵌套 begin 会清 pending）。
5. Resume 后自定义 reduce **必须仍已注册**（persist 的是名字，不是函数体）。
6. `begin_train_step` 已删除；误用会 `AssertionError`。
7. PP：非 last stage 不本地 finalize，依赖 broadcast；metrics 以 sync 后的 `final_values` 为准。
8. **`feature_store_enable=False` 时不要调 `get_ppo_feature_store()`**（会 raise）；关开关后单例被清掉。
9. EPO knobs **不要**写回 `ppo_config`；写 `task:`。Hydra 下 `config.task` 可能是 dict，用 `_task_get` 一类兼容读写。
10. DP 边角：若某些 DP rank **完全跳过**某 feature 的 `record`，而该 feature 用默认 `mean`，并集 reduce 仍可能按「有 pending 的 rank」聚合——当前 e2e（DP=1）未踩；多 DP 时尽量全员对称 record，或显式设计 reduce。

---



## 8. 与旧命名的关系


| 旧                                       | 新                                       |
| --------------------------------------- | --------------------------------------- |
| `train_extra_state` / `TrainExtraState` | `ppo_feature_store` / `PpoFeatureStore` |
| `train_extra_state.pt`                  | `ppo_feature_store.pt`（load 仍兼容旧文件名）    |
| core 内 EPO 专用 API                       | 已移除；EPO 是 task 侧 custom loss 的一个调用方     |
| `ppo.epo_*` / `epo_enable`              | 已移除；开关用 `feature_store_enable`，EPO knobs 在 `task.epo_*` |
| `_record_extra_metrics`                 | `report_extra_metrics`（mixin）           |


---



## 9. 相关文件速查


| 路径                                          | 说明                                      |
| ------------------------------------------- | --------------------------------------- |
| `gpatch_v4/core/ppo_feature_store/*`        | 实现                                      |
| `gpatch_v4/configs/ppo_config.py`           | `feature_store_enable`                  |
| `gpatch_v4/actor/grpo_train_actor.py`       | enable + interval + save/load           |
| `gpatch_v4/actor/grpo_async_train_actor.py` | async 同等接线                              |
| `gpatch_v4/actor/mixin.py`                  | `report_extra_metrics`                  |
| `tasks/math_rl_v4/epo_grpo_loss.py`         | EPO 用法（读 `task`）                        |
| `tasks/math_rl_v4/custom_advantage.py`      | advantage 多 reduce 用法                   |
| `memory/train_extra_state_generic_plan.md`  | 设计 plan / 实现记录                          |

