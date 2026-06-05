## 目录结构

```
gcore-dev/tools/convert_fp8/
├── qwen3_5/                        # Qwen3.5 / Qwen3.6 FP8 量化与测试
│   ├── quantize_hf_to_fp8.py       # 离线量化（FineGrainedFP8Config）
│   ├── qwen3_official_fp8_skips.py # 官方 skip list 生成
│   ├── qwen3_mtp_fp8_postprocess.py# MTP safetensors 后处理
│   ├── scripts/                    # 运行入口脚本
│   │   ├── run_quantize_qwen3_6_moe.sh # Qwen3.6 量化 / HF generate 测试
│   │   ├── launch_sglang_server.sh     # 启动 SGLang server
│   │   └── run_sglang_generate.sh      # 连接已启动 server 做 SGLang 推理测试
│   └── tests/                      # 单测与生成测试
├── logs/                           # 运行日志（远程执行时创建）
├── bf16_cast_fp8.py                # 原有（DeepSeek / gpatch Triton）
└── ...
```

## 量化（Qwen3.6-35B-A3B）

默认输入/输出（相对 `{gw_dir}`）：

- 输入：`hf-hub/Qwen/Qwen3.6-35B-A3B`（`{gw_dir}/hf-hub/Qwen/...`）
- 输出：`{gw_dir}/Qwen3.6-35B-A3B-FP8-official-skip`（不在 `hf-hub/` 下；绝对路径见下）

FP8 ckpt 目录（2026-05 起）：

`/mnt/ceph-hz1-csp/mm-base-plt2/user_xiaotaoliu/project/workspace3/gcore-dev/Qwen3.6-35B-A3B-FP8-official-skip`

远程 `{mrepo}` 等价路径：`/work/wepsdl/gcore-dev/Qwen3.6-35B-A3B-FP8-official-skip`

```bash
cd /work/wepsdl/gcore-dev
bash tools/convert_fp8/qwen3_5/scripts/run_quantize_qwen3_6_moe.sh --mode convert
# 或指定路径
bash tools/convert_fp8/qwen3_5/scripts/run_quantize_qwen3_6_moe.sh --mode convert \
  --input-hf-path /path/to/bf16 \
  --output-fp8-path /path/to/fp8
```

### Qwen3.5 / 3.6 官方对齐 skip 策略（默认）

`quantize_hf_to_fp8.py` 默认 `--skip-policy official`，按 HuggingFace 官方
`Qwen3.6-35B-A3B-FP8` 生成约 648 条显式 `modules_to_not_convert`（见
`qwen3_official_fp8_skips.py`）：

| 保持 BF16 | 量化为 FP8 |
|-----------|------------|
| ViT 全塔、`lm_head`、embed | MoE experts（含 `gate_up_proj` / `down_proj`） |
| MoE router（`mlp.gate`、`shared_expert_gate`） | full-attn 的 `q/k/v/o_proj` |
| 各层 `input/post_attention_layernorm` | linear-attn 大投影（`in_proj_qkv`、`in_proj_z`、`out_proj`） |
| linear-attn 小子模块（`in_proj_a/b`、`conv1d`、`norm` 等） | |
| full-attn 的 `q_norm` / `k_norm` | |
| MTP router / norm / `fc`（若存在） | MTP experts / attention projection（若存在） |

**注意**：旧版 `.*linear_attn` 会跳过整块 GatedDeltaNet（与官方相反）。如需旧行为：

```bash
python tools/convert_fp8/qwen3_5/quantize_hf_to_fp8.py ... --skip-policy legacy
```

也可从官方 FP8 ckpt 原样导入 skip 列表：

```bash
python tools/convert_fp8/qwen3_5/quantize_hf_to_fp8.py \
  --input-hf-path ... --output-fp8-path ... \
  --reference-fp8-config /work/wepsdl/gcore-dev/hf-hub/Qwen/Qwen3.6-35B-A3B-FP8
```

保存后会在 `config.json` 写入 `fmt: e4m3` 与完整 skip 列表。

### MTP 权重补齐

Transformers 当前 Qwen3.5 / 3.6 模型类会忽略顶层 `mtp.*` 权重，直接
`from_pretrained(..., FineGrainedFP8Config) -> save_pretrained(...)` 会丢掉 MTP。
本工具默认在保存主模型后执行 safetensors 后处理：

- 从 BF16 输入 ckpt 读取 `mtp.*`；
- 将 grouped expert 权重拆成官方 per-expert key；
- 对 MTP experts / attention projection 做 block-wise FP8；
- 将 router、norm、`mtp.fc`、`pre_fc_norm_*` 保持 BF16；
- 写入 `mtp.safetensors` 并更新 `model.safetensors.index.json`。

如确实不需要 MTP，可加：

```bash
python tools/convert_fp8/qwen3_5/quantize_hf_to_fp8.py ... --no-include-mtp
```

## HF 生成测试

```bash
LOG=tools/convert_fp8/logs/test_hf_generate_$(date +%Y%m%d_%H%M%S).log
python tools/convert_fp8/qwen3_5/tests/test_fp8_hf_generate.py \
  --fp8-path /work/wepsdl/gcore-dev/Qwen3.6-35B-A3B-FP8-official-skip \
  > "$LOG" 2>&1
```
