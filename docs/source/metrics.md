# Metrics

## _histogram metrics 

`TrainReporterSingleton.log_and_report` 里，key 以 `_histogram` 结尾的条目按 histogram 上报，其它 key 仍是标量。value 只接受下面两种，否则 `assert`。

### 格式

已经分好 bin：

```python
{
    "count": [n0, n1, ..., n_{B-1}],  # 每个 bin 的计数，长度 B
    "edges": [e0, e1, ..., e_B],      # bin 边界，长度必须是 B + 1
}
```

原始样本：

```python
[x0, x1, ..., xN]  # list 或 tuple
```

### 上报规则

| 后端 | `{count, edges}` | list / tuple |
|------|------------------|--------------|
| WandB | `wandb.Histogram(np_histogram=(count, edges))` | `wandb.Histogram(samples)`，默认 64 bins |
| TensorBoard | 跳过 | `add_histogram` |
| 文本 log | 跳过 | 跳过 |
