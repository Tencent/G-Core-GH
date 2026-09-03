# coding=utf-8
"""DSV4 model → disk rename table 单测（纯 CPU，<5s）。

对每个 disk-form leaf key 做 round-trip：
  HF forward (disk → model) → ``_model_key_to_disk_key`` (model → disk)
  → 必须等于原 disk key。

优先枚举真实 DSV4-Flash checkpoint 的 ``model.safetensors.index.json`` 全部 key
（~69k 条，过滤掉不需要 reverse 的 ``.scale`` / ``mtp.*``）；若 checkpoint 不
存在回退到硬编码 40-key 代表性 sample。

任何 mismatch 表示 ``_MODEL_TO_DISK_RENAMES`` 与 HF
``conversion_mapping["deepseek_v4"]`` forward 不同步。HF transformers 升级后
必跑此测试。
"""

import json
import os
import unittest
from packaging.version import Version
from importlib.metadata import version as pkg_version

# DSV4-Flash 标准 checkpoint 路径（与 test_tmp.py / test_save_load.py 一致）
_HF_INDEX = "hf-hub/deepseek-ai/DeepSeek-V4-Flash/model.safetensors.index.json"

# 硬编码 sample fallback：开发环境（无 checkpoint）也能跑
_FALLBACK_DISK_KEYS = [
    # top-level
    "embed.weight", "head.weight", "norm.weight",
    "hc_head_fn", "hc_head_base", "hc_head_scale",
    # 阶段 1: structural prefix + HC + norms
    "layers.0.attn_norm.weight", "layers.0.ffn_norm.weight",
    "layers.0.hc_attn_fn", "layers.0.hc_attn_base", "layers.0.hc_attn_scale",
    "layers.0.hc_ffn_fn", "layers.0.hc_ffn_base", "layers.0.hc_ffn_scale",
    # 阶段 2: attn outer leaves + sinks + q_norm
    "layers.0.attn.attn_sink",
    "layers.0.attn.wq_a.weight", "layers.0.attn.wq_b.weight",
    "layers.0.attn.wkv.weight", "layers.0.attn.wo_a.weight", "layers.0.attn.wo_b.weight",
    "layers.0.attn.q_norm.weight",
    # outer compressor (HCA + CSA outer)
    "layers.0.attn.compressor.wkv.weight", "layers.0.attn.compressor.wgate.weight",
    "layers.0.attn.compressor.norm.weight", "layers.0.attn.compressor.ape",
    # inner indexer (两层 disk 结构)
    "layers.0.attn.indexer.compressor.wkv.weight",
    "layers.0.attn.indexer.compressor.wgate.weight",
    "layers.0.attn.indexer.compressor.norm.weight",
    "layers.0.attn.indexer.compressor.ape",
    "layers.0.attn.indexer.wq_b.weight",
    "layers.0.attn.indexer.weights_proj.weight",
    # MoE 相关
    "layers.0.ffn.gate.weight", "layers.0.ffn.gate.bias", "layers.0.ffn.gate.tid2eid",
    "layers.0.ffn.experts.0.w1.weight", "layers.0.ffn.experts.0.w2.weight",
    "layers.0.ffn.experts.0.w3.weight", "layers.0.ffn.experts.255.w1.weight",
    "layers.0.ffn.shared_experts.w1.weight",
    "layers.0.ffn.shared_experts.w2.weight",
    "layers.0.ffn.shared_experts.w3.weight",
]


def _gather_disk_keys() -> tuple[list[str], str]:
    """优先读真实 checkpoint index.json，过滤后返回；否则回退到硬编码 sample。"""
    if os.path.isfile(_HF_INDEX):
        with open(_HF_INDEX) as f:
            weight_map = json.load(f)["weight_map"]
        # 过滤：.scale / .weight_scale_inv 是 FP8 quant scale，被 Fp8Dequantize 吸收，
        # save 路径不会产出；mtp.* HF DSV4 v5.8.1 无对应 module，被
        # _keys_to_ignore_on_load_unexpected 过滤掉，不会出现在我们的 state_dict。
        keys = sorted(
            k for k in weight_map
            if not k.endswith(".scale")
            and not k.endswith(".weight_scale_inv")
            and not k.startswith("mtp.")
        )
        return keys, f"enumerated {len(keys)} keys from {_HF_INDEX} (filtered)"
    return _FALLBACK_DISK_KEYS, f"fallback hardcoded sample ({len(_FALLBACK_DISK_KEYS)} keys)"


class TestRenameTableRoundTrip(unittest.TestCase):
    def test_round_trips_hf_forward(self):
        from transformers.conversion_mapping import _build_checkpoint_conversion_mapping
        from transformers.core_model_loading import WeightRenaming, rename_source_key

        from gpatch_v4.models.deepseek_v4.checkpoint import _model_key_to_disk_key

        base_convs = _build_checkpoint_conversion_mapping()["deepseek_v4"]
        renamings = [m for m in base_convs if isinstance(m, WeightRenaming)]
        # Adapt to transformers>=5.10.1 where ``weights_proj.weight`` is moved into ``scorer`` layer.
        if Version(pkg_version("transformers")) >= Version("5.10.1"):
            renamings.extend(
                [
                    WeightRenaming(
                        source_patterns=r"^(.*\.)?self_attn\.compressor\.indexer\.scorer\.weights_proj\.weight$",
                        target_patterns=r"\1self_attn.compressor.indexer.weights_proj.weight",
                    ),
                ]
            )

        disk_keys, source = _gather_disk_keys()
        print(f"[rename_table_test] {source}")

        failures: list[str] = []
        for disk_key in disk_keys:
            # forward (disk → model) via HF
            model_key, _ = rename_source_key(disk_key, renamings, [], "model", {})
            model_key = model_key.removeprefix("model.")
            # reverse (model → disk) via 我们的 table
            roundtrip = _model_key_to_disk_key(model_key)
            if roundtrip != disk_key:
                failures.append(
                    f"{disk_key!r} → {model_key!r} → {roundtrip!r}"
                )

        head = "\n".join(f"  {f}" for f in failures[:20])
        tail = f"\n  ... (+{len(failures) - 20} more)" if len(failures) > 20 else ""
        self.assertFalse(
            failures,
            msg=f"rename table 与 HF forward 不同步 ({len(failures)}/{len(disk_keys)} keys):\n"
                + head + tail,
        )


if __name__ == "__main__":
    unittest.main()
