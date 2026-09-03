"""Fused dyn-cp layout / compare 单测（无需多卡）。"""
import unittest

import torch

from gpatch_v4.utils.dynamic_cp_utils import (
    _REROUTE_ALIGN,
    _plan_fused_dest_slice,
    compare_reroute_sample_dicts,
)


class TestFusedRerouteLayout(unittest.TestCase):

    def test_slice_total_64_aligned_with_odd_bf16_tail(self):
        data_keys = ["tokens", "zzz_feat"]
        key_dtypes = {"tokens": torch.int64, "zzz_feat": torch.bfloat16}
        key_numels = {"tokens": 4, "zzz_feat": 3}
        entries, _meta, total = _plan_fused_dest_slice(data_keys, key_numels, key_dtypes)
        self.assertEqual(total % _REROUTE_ALIGN, 0)
        for entry in entries:
            self.assertEqual(entry["byte_offset"] % _REROUTE_ALIGN, 0)

        # 两 slice 拼接后，第二片 tokens 的绝对 offset 仍按 8 对齐。
        _, _, total2 = _plan_fused_dest_slice(data_keys, key_numels, key_dtypes)
        tok_off = next(e["byte_offset"] for e in entries if e["key"] == "tokens")
        self.assertEqual((total + tok_off) % 8, 0)
        self.assertEqual(total2, total)

    def test_compare_reroute_sample_dicts(self):
        left = {
            0: {
                "tokens": torch.tensor([1, 2], dtype=torch.int64),
                "loss_mask": torch.tensor([1.0, 0.0], dtype=torch.float32),
            }
        }
        right = {
            0: {
                "tokens": torch.tensor([1, 2], dtype=torch.int64),
                "loss_mask": torch.tensor([1.0, 0.0], dtype=torch.float32),
            }
        }
        compare_reroute_sample_dicts(left, right)
        right_bad = {
            0: {
                "tokens": torch.tensor([9, 2], dtype=torch.int64),
                "loss_mask": torch.tensor([1.0, 0.0], dtype=torch.float32),
            }
        }
        with self.assertRaises(AssertionError):
            compare_reroute_sample_dicts(left, right_bad)
