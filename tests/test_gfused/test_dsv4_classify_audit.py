# coding=utf-8
"""DSV4 `_classify_for_save` 与 DSV4-Flash 磁盘事实对齐审计（纯 CPU，<30s）。

Stage 1.5：在 quant save 路径全 43L 跑 Stage 2 之前必跑。

直接读真实 DSV4-Flash checkpoint 的所有 shard header，按每个 `.weight` 键的
on-disk dtype + shape 验证：

  1. ``_classify_for_save`` 的输出与磁盘真相一致——
     int8 → fp4_expert / float8_e4m3fn → fp8_e4m3 /
     bfloat16 → bf16_passthrough / float32 → f32_passthrough。
     任何错配都说明 ``_FP8_DISK_KEY_PATTERNS`` / ``_F32_DISK_KEY_PATTERNS``
     与 DSV4-Flash 实际结构不一致——bf16 路径只是文件大几个字节，**quant
     路径会让 vanilla `from_pretrained` shape mismatch 报错**。
  2. fp4_expert keys 满足 FP4 quantizer 的 shape 约束（last-dim % 32 + 偶数）。
  3. fp8_e4m3 keys 满足 FP8 quantizer 的 shape 约束（两维 % 128）。

任何 mismatch 都是 quant save 上 Stage 2 之前 MUST FIX 的真 bug。

checkpoint 不存在则 SKIP——本测试无 hardcoded fallback（对齐审计无意义否则）。
"""

import json
import os
import struct
import unittest

# DSV4-Flash 标准 checkpoint 路径（与 test_save_load.py / test_rename_table.py 一致）
_HF_DIR = "hf-hub/deepseek-ai/DeepSeek-V4-Flash"
_HF_INDEX = os.path.join(_HF_DIR, "model.safetensors.index.json")


def _read_safetensors_header(shard_path: str) -> dict:
    """读 safetensors shard 文件头（不加载 weights）。

    safetensors 文件格式：[8 字节 little-endian header_size][header_size 字节
    JSON header][raw tensor data]。header JSON 形如
    ``{"key": {"dtype": "F8_E4M3", "shape": [...], "data_offsets": [...]}}``。
    """
    with open(shard_path, "rb") as f:
        (header_size,) = struct.unpack("<Q", f.read(8))
        header_bytes = f.read(header_size)
    return json.loads(header_bytes)


def _gather_disk_metadata() -> dict[str, dict]:
    """枚举 DSV4-Flash 所有 `.weight` 键的 (dtype, shape)。

    跳过 `.scale` / `.weight_scale_inv` (FP8 quant scale，被 Fp8Dequantize 吸收)
    与 `mtp.*` (HF DSV4 v5.8.1 无 MTP module，被 _keys_to_ignore_on_load_unexpected 过滤)。
    """
    with open(_HF_INDEX) as f:
        weight_map = json.load(f)["weight_map"]

    # 按 shard 分组，避免对同一 shard 多次读 header
    shard_to_keys: dict[str, list[str]] = {}
    for k, shard in weight_map.items():
        if k.endswith(".scale") or k.endswith(".weight_scale_inv") or k.startswith("mtp."):
            continue
        shard_to_keys.setdefault(shard, []).append(k)

    metadata: dict[str, dict] = {}
    for shard, keys in shard_to_keys.items():
        header = _read_safetensors_header(os.path.join(_HF_DIR, shard))
        for k in keys:
            entry = header.get(k)
            assert entry is not None, f"key {k!r} not in shard header {shard}"
            metadata[k] = {"dtype": entry["dtype"], "shape": entry["shape"]}
    return metadata


# safetensors 头部 dtype 字符串 → (期望 _classify_for_save 输出, save 时分类器看到的 dtype)
#
# 关键：feed dtype 必须与 save 路径在 wave-B dispatch 处实际看到的 `mv.dtype` 一致。
# 浮点 weight 经过 fp32 master + apply_hp mp_policy 后到 wave-B 都是 fp32，无论
# disk 上是 I8 (packed FP4) / F8_E4M3 / BF16 / F32。但 int 持久 buffer (tid2eid)
# 没有 master 副本，wave-B 看到的就是 int64 本身。
#
# 喂错 dtype 会触发分类器入口的 `if not t.dtype.is_floating_point: return "int_passthrough"`
# 短路：例如 I8 喂 int8 → 全 expert 误判 int_passthrough；I64 喂 fp32 →
# tid2eid 误判 bf16_passthrough。
_DTYPE_TO_AUDIT: dict[str, tuple[str, "torch.dtype"]] = {}


def _build_audit_map() -> None:
    """延迟构造 audit map，避免模块顶部 import torch（CPU 单测环境兼容）。"""
    import torch
    _DTYPE_TO_AUDIT.update({
        "I8":      ("fp4_expert",       torch.float32),  # packed FP4 expert weight, save-time master = fp32
        "F8_E4M3": ("fp8_e4m3",         torch.float32),  # fine-grained FP8 dense, save-time master = fp32
        "BF16":    ("bf16_passthrough", torch.float32),  # bf16 norm/embed, save-time master = fp32
        "F32":     ("f32_passthrough",  torch.float32),  # fp32 norm/sink, save-time master = fp32
        "I64":     ("int_passthrough",  torch.int64),    # int64 persistent buffer (tid2eid), no master cast
    })


class TestClassifyAuditAgainstDsv4Flash(unittest.TestCase):
    """`_classify_for_save` 输出必须 100% 对齐 DSV4-Flash 磁盘事实。"""

    @classmethod
    def setUpClass(cls):
        cls.metadata = _gather_disk_metadata()
        print(f"\n[audit] enumerated {len(cls.metadata)} weight keys from {_HF_INDEX}")

    def test_classifier_matches_disk_truth(self):
        """每个 weight/buffer 键的 _classify_for_save 输出必须与磁盘 dtype 一致。"""
        import torch
        from gpatch_v4.models.deepseek_v4.checkpoint import _classify_for_save

        _build_audit_map()
        # 缓存每种 audit dtype 的 fake tensor，避免在 ~34k 次循环里反复构造
        fake_cache: dict[torch.dtype, torch.Tensor] = {
            d: torch.empty(0, dtype=d) for _, d in _DTYPE_TO_AUDIT.values()
        }

        mismatches: list[tuple[str, str, str, str]] = []  # (key, disk_dtype, expected_cls, actual_cls)
        for key, info in self.metadata.items():
            disk_dtype_str = info["dtype"]
            audit_entry = _DTYPE_TO_AUDIT.get(disk_dtype_str)
            self.assertIsNotNone(
                audit_entry,
                f"unexpected disk dtype {disk_dtype_str!r} for {key!r}; "
                f"add to _DTYPE_TO_AUDIT or investigate",
            )
            expected_cls, audit_dtype = audit_entry
            actual_cls = _classify_for_save(key, fake_cache[audit_dtype])
            if actual_cls != expected_cls:
                mismatches.append((key, disk_dtype_str, expected_cls, actual_cls))

        if mismatches:
            preview = "\n".join(
                f"  {k!r}  disk={dd}  expected={e}  actual={a}"
                for k, dd, e, a in mismatches[:20]
            )
            self.fail(
                f"\n_classify_for_save mismatches DSV4-Flash disk truth on "
                f"{len(mismatches)} keys (first 20):\n{preview}\n\n"
                f"Fix _FP8_DISK_KEY_PATTERNS / _F32_DISK_KEY_PATTERNS in "
                f"gpatch_v4/models/deepseek_v4/checkpoint.py before running "
                f"quant save Stage 2."
            )

    def test_fp4_shape_divisibility(self):
        """所有 fp4_expert keys 必须满足 FP4 quantizer 的 shape 约束（unpacked last-dim % 32）。"""
        bad: list[tuple[str, list[int]]] = []
        for key, info in self.metadata.items():
            if info["dtype"] != "I8":
                continue
            shape = info["shape"]
            self.assertGreaterEqual(len(shape), 1, f"{key!r} unexpected scalar")
            n = shape[-1]
            # ⚠️ DSV4-Flash 磁盘上 FP4 expert 是 packed I8（last-dim 已折半）。
            # quant_fp4_e2m1_scale_e8m0_packed 的输入是未 packed 的 fp32 weight，
            # last-dim 是 packed 后的 ×2，所以 disk last-dim ×2 必须 % 32 == 0
            # 等价于 disk last-dim % 16 == 0。`× 2` 天然是偶数，无需额外 check。
            if (n * 2) % 32 != 0:
                bad.append((key, shape))
        self.assertFalse(
            bad,
            f"fp4_expert keys violate FP4 shape constraint (unpacked last-dim must "
            f"be %32==0): {bad[:10]}",
        )

    def test_fp8_shape_divisibility(self):
        """所有 fp8_e4m3 keys 必须满足 FP8 quantizer 的 shape 约束（两维 % 128）。"""
        bad: list[tuple[str, list[int]]] = []
        for key, info in self.metadata.items():
            if info["dtype"] != "F8_E4M3":
                continue
            shape = info["shape"]
            if len(shape) < 2:
                bad.append((key, shape))
                continue
            m, n = shape[-2], shape[-1]
            if m % 128 != 0 or n % 128 != 0:
                bad.append((key, shape))
        self.assertFalse(
            bad,
            f"fp8_e4m3 keys violate FP8 shape constraint (last two dims must "
            f"be %128==0): {bad[:10]}",
        )


if __name__ == "__main__":
    unittest.main()
