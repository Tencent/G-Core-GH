"""Unit tests for Qwen3 FP8 quantization wrapper helpers."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

from quantize_hf_to_fp8 import copy_processor_configs


class QuantizeHfToFp8Test(unittest.TestCase):
    def test_copy_processor_configs(self):
        with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory(
        ) as output_dir:
            expected = {
                "preprocessor_config.json": {
                    "image_processor_type": "Qwen3ImageProcessor"
                },
                "video_preprocessor_config.json": {
                    "video_processor_type": "Qwen3VideoProcessor"
                },
            }
            for name, payload in expected.items():
                with open(os.path.join(input_dir, name), "w") as f:
                    json.dump(payload, f)
            with open(os.path.join(input_dir, "config.json"), "w") as f:
                json.dump({"ignored": True}, f)

            copied = copy_processor_configs(input_dir, output_dir)

            self.assertEqual(copied, sorted(expected))
            for name, payload in expected.items():
                with open(os.path.join(output_dir, name)) as f:
                    self.assertEqual(json.load(f), payload)
            self.assertFalse(os.path.exists(os.path.join(output_dir, "config.json")))


if __name__ == "__main__":
    unittest.main()
