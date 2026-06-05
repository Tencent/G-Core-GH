# patch — 第三方库补丁

对第三方库的 monkey-patch，解决与分布式训练环境的兼容性问题。

## 文件说明

| 文件 | 说明 |
|------|------|
| `sglang_patch.py` | `sglang_hack()` — 清理 SGLang 与 `torch.distributed` 冲突的环境变量，调用 `monkey_patch_torch_reductions()` |
