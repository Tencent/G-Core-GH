"""Forbidden token ids from real Qwen checkpoints under hf-hub/Qwen."""
import json
import os
from types import SimpleNamespace
from unittest import TestCase

from megatron_datasets.utils import build_forbidden_token_ids

_HF_QWEN_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "hf-hub", "Qwen")
)

_QWEN3_VL = "Qwen3-VL-2B-Instruct"
_QWEN3_5 = "Qwen3.5-4B"
_QWEN3_5_MOE = "Qwen3.5-35B-A3B"
_QWEN3_OMNI_CANDIDATES = (
    "Qwen3-Omni-30B-A3B-Instruct",
    "Qwen3-Omni-30B-A3B",
    "Qwen3-Omni-Moe-30B-A3B-Instruct",
)


def _ckpt_dir(*names: str) -> str | None:
    for name in names:
        path = os.path.join(_HF_QWEN_ROOT, name)
        if os.path.isfile(os.path.join(path, "config.json")):
            return path
    return None


def _dict_to_ns(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _dict_to_ns(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_dict_to_ns(item) for item in value]
    return value


def _load_hf_config(ckpt_dir: str):
    with open(os.path.join(ckpt_dir, "config.json"), encoding="utf-8") as handle:
        return _dict_to_ns(json.load(handle))


class TestBuildForbiddenTokenIds(TestCase):
    def _require_ckpt(self, *names: str) -> str:
        path = _ckpt_dir(*names)
        if path is None:
            self.skipTest(f"checkpoint not found under {_HF_QWEN_ROOT}: {names}")
        return path

    def test_qwen3_vl_checkpoint(self) -> None:
        cfg = _load_hf_config(self._require_ckpt(_QWEN3_VL))
        got = build_forbidden_token_ids(cfg)
        self.assertEqual(getattr(cfg, "model_type", None), "qwen3_vl")
        self.assertIn(cfg.image_token_id, got)
        self.assertIn(cfg.video_token_id, got)
        self.assertIn(cfg.vision_start_token_id, got)
        self.assertIn(cfg.vision_end_token_id, got)

    def test_qwen3_5_checkpoint(self) -> None:
        cfg = _load_hf_config(self._require_ckpt(_QWEN3_5))
        got = build_forbidden_token_ids(cfg)
        self.assertEqual(getattr(cfg, "model_type", None), "qwen3_5")
        self.assertIn(cfg.image_token_id, got)
        self.assertIn(cfg.video_token_id, got)
        self.assertIn(cfg.vision_start_token_id, got)
        self.assertIn(cfg.vision_end_token_id, got)

    def test_qwen3_5_moe_checkpoint(self) -> None:
        cfg = _load_hf_config(self._require_ckpt(_QWEN3_5_MOE))
        got = build_forbidden_token_ids(cfg)
        self.assertEqual(getattr(cfg, "model_type", None), "qwen3_5_moe")
        self.assertIn(cfg.image_token_id, got)
        self.assertIn(cfg.vision_start_token_id, got)

    def test_qwen3_omni_moe_checkpoint(self) -> None:
        cfg = _load_hf_config(self._require_ckpt(*_QWEN3_OMNI_CANDIDATES))
        got = build_forbidden_token_ids(cfg)
        thinker = getattr(cfg, "thinker_config", None)
        talker = getattr(cfg, "talker_config", None)
        self.assertIsNotNone(thinker, "expected nested thinker_config on Omni checkpoint")
        self.assertIn(thinker.image_token_id, got)
        self.assertIn(thinker.video_token_id, got)
        self.assertIn(thinker.audio_token_id, got)
        self.assertIn(thinker.audio_start_token_id, got)
        if talker is not None and getattr(talker, "vision_start_token_id", None) is not None:
            self.assertIn(talker.vision_start_token_id, got)
        self.assertNotIn(getattr(cfg, "im_start_token_id", None), got)
        self.assertNotIn(getattr(cfg, "im_end_token_id", None), got)

    def test_unwrapped_omni_thinker_misses_vision_start(self) -> None:
        cfg = _load_hf_config(self._require_ckpt(*_QWEN3_OMNI_CANDIDATES))
        thinker = getattr(cfg, "thinker_config", None)
        talker = getattr(cfg, "talker_config", None)
        if thinker is None or talker is None:
            self.skipTest("Omni checkpoint missing thinker/talker split")
        vision_start = getattr(talker, "vision_start_token_id", None)
        if vision_start is None:
            self.skipTest("talker_config has no vision_start_token_id")
        got = build_forbidden_token_ids(thinker)
        self.assertNotIn(vision_start, got)
        self.assertIn(thinker.audio_token_id, got)

    def test_none_and_empty_config(self) -> None:
        self.assertEqual(build_forbidden_token_ids(None), [])
        self.assertEqual(build_forbidden_token_ids(SimpleNamespace()), [])

    def test_skips_attributeerror_sentinel(self) -> None:
        hf_config = SimpleNamespace(
            image_token_id=151655,
            audio_end_token_id=AttributeError(),
        )
        self.assertEqual(build_forbidden_token_ids(hf_config), [151655])

    def test_property_empty_config_becomes_none(self) -> None:
        # Qwen3VLPrepareDataForward.forbidden_token_ids uses ``or None``.
        self.assertIsNone(build_forbidden_token_ids(None) or None)
        self.assertIsNone(build_forbidden_token_ids(SimpleNamespace()) or None)
