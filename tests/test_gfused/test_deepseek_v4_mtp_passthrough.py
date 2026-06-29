# coding=utf-8
"""Unit test for MTP passthrough in _rank0_finalize (pure CPU, <5s).

Verifies that when ``preserve_mtp=True``, MTP tensors from the source
checkpoint are copied into the saved directory and included in the output
``model.safetensors.index.json``.
"""

import json
import os
import tempfile
import unittest

import safetensors.torch
import torch


def _make_fake_source_ckpt(src_dir: str) -> dict[str, str]:
    """Create a minimal source checkpoint with MTP keys."""
    mtp_tensors = {
        "mtp.0.norm.weight": torch.randn(128),
        "mtp.0.ffn_norm.weight": torch.randn(128),
        "mtp.0.attn.wq_a.weight": torch.randn(128, 64),
        "mtp.0.attn.wkv.weight": torch.randn(128, 64),
        "mtp.0.ffn.shared_experts.w1.weight": torch.randn(256, 128),
        "mtp.0.ffn.shared_experts.w2.weight": torch.randn(128, 256),
        "mtp.0.ffn.gate.weight": torch.randn(8, 128),
        "mtp.0.ffn.gate.bias": torch.randn(8),
    }
    shard_name = "model-00002-of-00002.safetensors"
    safetensors.torch.save_file(mtp_tensors, os.path.join(src_dir, shard_name))

    non_mtp_tensors = {"layers.0.attn_norm.weight": torch.randn(128)}
    non_mtp_shard = "model-00001-of-00002.safetensors"
    safetensors.torch.save_file(
        non_mtp_tensors, os.path.join(src_dir, non_mtp_shard)
    )

    weight_map = {}
    for k in non_mtp_tensors:
        weight_map[k] = non_mtp_shard
    for k in mtp_tensors:
        weight_map[k] = shard_name
    index = {"metadata": {"total_size": 0}, "weight_map": weight_map}
    with open(os.path.join(src_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)

    return mtp_tensors


def _make_fake_model_shards(save_dir: str) -> tuple[list[dict], str]:
    """Create fake per-rank model shards as produced by the wave loop."""
    placeholder_suffix = "of-XXXXX.safetensors"

    model_tensors_r0 = {
        "embed.weight": torch.randn(100, 128),
        "norm.weight": torch.randn(128),
    }
    shard_name_r0 = f"model-rank00-00001-{placeholder_suffix}"
    safetensors.torch.save_file(
        model_tensors_r0, os.path.join(save_dir, shard_name_r0)
    )

    gather_list = [
        {
            "rank": 0,
            "shard_count": 1,
            "tensor_to_filename": {k: shard_name_r0 for k in model_tensors_r0},
            "total_size_bytes": sum(
                t.element_size() * t.nelement() for t in model_tensors_r0.values()
            ),
        },
    ]
    return gather_list, placeholder_suffix


class _FakeModel:
    """Minimal mock standing in for nn.Module in _rank0_finalize."""

    class _FakeConfig:
        def save_pretrained(self, path):
            cfg = {"model_type": "deepseek_v4", "hidden_size": 128}
            with open(os.path.join(path, "config.json"), "w") as f:
                json.dump(cfg, f)

    class _FakeGenConfig:
        def save_pretrained(self, path):
            with open(os.path.join(path, "generation_config.json"), "w") as f:
                json.dump({}, f)

    config = _FakeConfig()
    generation_config = _FakeGenConfig()

    def can_generate(self):
        return True


class TestMtpPassthrough(unittest.TestCase):
    def test_mtp_keys_in_saved_index(self):
        from gpatch_v4.models.deepseek_v4.checkpoint import _rank0_finalize

        tmpdir = "/tmp/test_mtp_passthrough"
        src_dir = os.path.join(tmpdir, "src_ckpt")
        save_dir = os.path.join(tmpdir, "save_ckpt")
        # Clean up from previous runs
        import shutil
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
        os.makedirs(src_dir)
        os.makedirs(save_dir)

        if True:

            # Create fake tokenizer files required by _rank0_finalize
            for fname in ("tokenizer.json", "tokenizer_config.json"):
                with open(os.path.join(src_dir, fname), "w") as f:
                    json.dump({}, f)

            mtp_tensors = _make_fake_source_ckpt(src_dir)
            gather_list, placeholder_suffix = _make_fake_model_shards(save_dir)

            _rank0_finalize(
                self=_FakeModel(),
                save_path=save_dir,
                orig_ckpt_dir=src_dir,
                gather_list=gather_list,
                placeholder_shard_suffix=placeholder_suffix,
                world_size=1,
                max_shard_size=5 * 1024**3,
                dtype_format="quantized",
                preserve_mtp=True,
            )

            # Verify index.json exists and contains MTP keys
            index_path = os.path.join(save_dir, "model.safetensors.index.json")
            self.assertTrue(os.path.isfile(index_path))
            with open(index_path) as f:
                saved_index = json.load(f)

            weight_map = saved_index["weight_map"]
            saved_mtp_keys = {k for k in weight_map if k.startswith("mtp.")}
            expected_mtp_keys = set(mtp_tensors.keys())
            self.assertEqual(saved_mtp_keys, expected_mtp_keys)

            # Verify model keys are also present
            self.assertIn("embed.weight", weight_map)
            self.assertIn("norm.weight", weight_map)

            # Verify shard filenames use correct total (model + MTP)
            shard_names = set(weight_map.values())
            for sname in shard_names:
                self.assertRegex(sname, r"model-\d{5}-of-\d{5}\.safetensors")
                # Total should be 2 (1 model shard + 1 MTP shard)
                self.assertIn("-of-00002", sname)

            # Verify MTP tensors are byte-for-byte correct
            mtp_shard_name = weight_map["mtp.0.norm.weight"]
            mtp_shard_path = os.path.join(save_dir, mtp_shard_name)
            self.assertTrue(os.path.isfile(mtp_shard_path))
            loaded = safetensors.torch.load_file(mtp_shard_path)
            for key, expected in mtp_tensors.items():
                self.assertIn(key, loaded)
                torch.testing.assert_close(loaded[key], expected)

    def test_preserve_mtp_false_skips(self):
        from gpatch_v4.models.deepseek_v4.checkpoint import _rank0_finalize

        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "src_ckpt")
            save_dir = os.path.join(tmpdir, "save_ckpt")
            os.makedirs(src_dir)
            os.makedirs(save_dir)

            for fname in ("tokenizer.json", "tokenizer_config.json"):
                with open(os.path.join(src_dir, fname), "w") as f:
                    json.dump({}, f)

            _make_fake_source_ckpt(src_dir)
            gather_list, placeholder_suffix = _make_fake_model_shards(save_dir)

            _rank0_finalize(
                self=_FakeModel(),
                save_path=save_dir,
                orig_ckpt_dir=src_dir,
                gather_list=gather_list,
                placeholder_shard_suffix=placeholder_suffix,
                world_size=1,
                max_shard_size=5 * 1024**3,
                dtype_format="quantized",
                preserve_mtp=False,
            )

            index_path = os.path.join(save_dir, "model.safetensors.index.json")
            with open(index_path) as f:
                saved_index = json.load(f)

            weight_map = saved_index["weight_map"]
            saved_mtp_keys = {k for k in weight_map if k.startswith("mtp.")}
            self.assertEqual(saved_mtp_keys, set())
            # Only 1 model shard, so total=1
            for sname in weight_map.values():
                self.assertIn("-of-00001", sname)

    def test_no_source_index_warns_gracefully(self):
        from gpatch_v4.models.deepseek_v4.checkpoint import _rank0_finalize

        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "src_ckpt")
            save_dir = os.path.join(tmpdir, "save_ckpt")
            os.makedirs(src_dir)
            os.makedirs(save_dir)

            for fname in ("tokenizer.json", "tokenizer_config.json"):
                with open(os.path.join(src_dir, fname), "w") as f:
                    json.dump({}, f)

            # No model.safetensors.index.json in src_dir
            gather_list, placeholder_suffix = _make_fake_model_shards(save_dir)

            # Should not raise
            _rank0_finalize(
                self=_FakeModel(),
                save_path=save_dir,
                orig_ckpt_dir=src_dir,
                gather_list=gather_list,
                placeholder_shard_suffix=placeholder_suffix,
                world_size=1,
                max_shard_size=5 * 1024**3,
                dtype_format="quantized",
                preserve_mtp=True,
            )

            index_path = os.path.join(save_dir, "model.safetensors.index.json")
            with open(index_path) as f:
                saved_index = json.load(f)
            weight_map = saved_index["weight_map"]
            saved_mtp_keys = {k for k in weight_map if k.startswith("mtp.")}
            self.assertEqual(saved_mtp_keys, set())


if __name__ == "__main__":
    unittest.main()
