# coding=utf-8
# copyright (c) 2026 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com, yea@tencent.com
"""Tests for the MoE offline repermute pipeline orchestration."""
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from tools.moe_offline_repermute.repermute_pipeline import (
    MoeOfflineRepermuteConfig,
    resolve_model_shape,
    run_moe_offline_repermute,
)

# Minimal flat MoE config accepted by transformers' AutoConfig. ``qwen3_moe`` is
# the flat (non-VL) registered model type in the remote env's transformers.
_FLAT_CONFIG = {
    "model_type": "qwen3_moe",
    "num_hidden_layers": 6,
    "num_experts": 16,
    "hidden_size": 32,
    "intermediate_size": 64,
    "vocab_size": 100,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "moe_intermediate_size": 32,
    "num_experts_per_tok": 2,
}

_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
_QWEN36_VL_CONFIG_JSON = os.path.join(_FIXTURE_DIR, "qwen3_6_35b_a3b_config.json")


class TestMoeOfflineRepermutePipeline(unittest.TestCase):
    def test_pipeline_calls_steps_in_order_with_derived_paths(self) -> None:
        calls = []

        def record(name):
            def _inner(**kwargs):
                calls.append((name, kwargs))

            return _inner

        with mock.patch(
            "tools.moe_offline_repermute.repermute_pipeline.build_routing_map",
            side_effect=record("build"),
        ), mock.patch(
            "tools.moe_offline_repermute.repermute_pipeline.repermute_hf_ckpt",
            side_effect=record("repermute"),
        ), mock.patch(
            "tools.moe_offline_repermute.repermute_pipeline.verify_equivalence",
            side_effect=record("verify"),
        ), mock.patch(
            "tools.moe_offline_repermute.repermute_pipeline.visualize_repermute",
            side_effect=record("viz"),
        ):
            result = run_moe_offline_repermute(
                MoeOfflineRepermuteConfig(
                    source_dump_dir="/tmp/source_dump",
                    src_ckpt_dir="/tmp/src_ckpt",
                    dst_ckpt_dir="/tmp/dst_ckpt",
                    work_dir="/tmp/work",
                    out_dir=None,
                    num_layers=2,
                    num_experts=8,
                    ep_sizes=(2, 4),
                    verify_full_forward=False,
                    sample_type="text",
                    force_refresh_aux=True,
                )
            )

        routing_map_path = os.path.join("/tmp/work", "routing_map.json")
        imbalance_curve_path = os.path.join("/tmp/work", "imbalance_curve.png")
        viz_out_dir = os.path.join("/tmp/work", "viz")

        self.assertEqual(
            [name for name, _ in calls], ["build", "repermute", "repermute", "verify", "viz"]
        )
        self.assertEqual(
            calls[0][1],
            {
                "source_dir": "/tmp/source_dump",
                "output": routing_map_path,
                "imbalance_curve": imbalance_curve_path,
                "num_layers": 2,
                "num_experts": 8,
            },
        )
        self.assertTrue(calls[1][1]["dry_run"])
        self.assertFalse(calls[2][1]["dry_run"])
        self.assertEqual(calls[1][1]["routing_map_path"], routing_map_path)
        self.assertEqual(calls[2][1]["routing_map_path"], routing_map_path)
        self.assertFalse(calls[3][1]["verify_full_forward"])
        self.assertEqual(calls[4][1]["out_dir"], viz_out_dir)
        self.assertEqual(calls[4][1]["ep_sizes"], (2, 4))
        self.assertEqual(result.routing_map_path, routing_map_path)
        self.assertEqual(result.imbalance_curve_path, imbalance_curve_path)
        self.assertEqual(result.dst_ckpt_dir, "/tmp/dst_ckpt")
        self.assertEqual(result.viz_out_dir, viz_out_dir)

    def test_resolve_model_shape_reads_flat_config(self) -> None:
        """Flat (non-VL) checkpoints: ``get_text_config()`` returns ``self``."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config.json"), "w") as fp:
                json.dump(_FLAT_CONFIG, fp)
            self.assertEqual(resolve_model_shape(src_ckpt_dir=tmpdir), (6, 16))

    def test_resolve_model_shape_unwraps_qwen_vl_text_config(self) -> None:
        """Regression: Qwen3.6-VL keeps MoE fields under ``text_config``.

        Loads the real ``Qwen3.6-35B-A3B/config.json`` fixture (weights not
        needed) so future Qwen VL schema drift breaks this test.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.copy(_QWEN36_VL_CONFIG_JSON, os.path.join(tmpdir, "config.json"))
            num_layers, num_experts = resolve_model_shape(src_ckpt_dir=tmpdir)
            self.assertEqual(num_layers, 40)
            self.assertEqual(num_experts, 256)

    def test_resolve_model_shape_allows_cli_override_without_reading_config(self, ) -> None:
        """Both overrides given: never touches the filesystem or transformers."""
        self.assertEqual(
            resolve_model_shape(
                src_ckpt_dir="/nonexistent/dir",
                num_layers=2,
                num_experts=8,
            ),
            (2, 8),
        )
