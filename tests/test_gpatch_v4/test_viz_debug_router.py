# coding=utf-8
# copyright (c) 2026 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com, yea@tencent.com
"""Tests for ``tools.moe_offline_repermute.viz_debug_router``."""
import os
import tempfile
import unittest

import numpy as np

from tools.moe_offline_repermute.viz_debug_router import (
    dst_to_old,
    parse_router_blocks,
    split_src_dst,
    top_k_by_freq,
)


def _mk_tensor_block(rows, device: int = 0) -> str:
    """Format a list of lists of ints as a DEBUG router select block."""
    lines = []
    for i, row in enumerate(rows):
        inner = ", ".join(f"{x:3d}" for x in row)
        if i == 0 and len(rows) == 1:
            lines.append(
                f"DEBUG router select router_indices=tensor([[{inner}]], "
                f"device='cuda:{device}')"
            )
        elif i == 0:
            lines.append(f"DEBUG router select router_indices=tensor([[{inner}],")
        elif i == len(rows) - 1:
            lines.append(f"        [{inner}]], device='cuda:{device}')")
        else:
            lines.append(f"        [{inner}],")
    return "\n".join(lines)


class TestVizDebugRouter(unittest.TestCase):
    def test_parse_router_blocks_extracts_tensors(self) -> None:
        rows = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]  # T=3, K=4
        log_text = (
            "some preamble\n"
            "Loading weights: 100%|#####| 10/10\n"
            f"{_mk_tensor_block(rows)}\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "fake.log")
            with open(log_path, "w") as fp:
                fp.write(log_text)
            blocks, line_nos, anchor = parse_router_blocks(log_path)
        self.assertEqual(blocks.shape, (1, 3, 4))
        np.testing.assert_array_equal(blocks[0], np.asarray(rows, dtype=np.int64))
        self.assertEqual(len(line_nos), 1)
        self.assertEqual(anchor, 2)  # line 2 has 'Loading weights:'

    def test_parse_router_blocks_errors_on_missing_anchor(self) -> None:
        # A block but no `Loading weights:` anchor -- assertion should fire.
        rows = [[0, 1, 2, 3]]
        log_text = f"preamble\n{_mk_tensor_block(rows)}\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "fake.log")
            with open(log_path, "w") as fp:
                fp.write(log_text)
            with self.assertRaisesRegex(AssertionError, "Loading weights"):
                parse_router_blocks(log_path)

    def test_split_src_dst_uses_loading_weights_anchor(self) -> None:
        # 2 src blocks, then `Loading weights:`, then 2 dst blocks.
        # num_layers=2 so split should be exactly 2+2.
        rows_a = [[0, 1, 2, 3], [4, 5, 6, 7]]
        rows_b = [[10, 11, 12, 13], [14, 15, 16, 17]]
        rows_c = [[20, 21, 22, 23], [24, 25, 26, 27]]
        rows_d = [[30, 31, 32, 33], [34, 35, 36, 37]]
        log_text = "\n".join(
            [
                "preamble",
                _mk_tensor_block(rows_a),
                _mk_tensor_block(rows_b),
                "Loading weights: 100%|#####| 100/100",
                _mk_tensor_block(rows_c),
                _mk_tensor_block(rows_d),
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "fake.log")
            with open(log_path, "w") as fp:
                fp.write(log_text)
            blocks, line_nos, anchor = parse_router_blocks(log_path)
            src, dst = split_src_dst(blocks, line_nos, anchor, num_layers=2)
        self.assertEqual(src.shape, (2, 2, 4))
        self.assertEqual(dst.shape, (2, 2, 4))
        np.testing.assert_array_equal(src[0], np.asarray(rows_a))
        np.testing.assert_array_equal(src[1], np.asarray(rows_b))
        np.testing.assert_array_equal(dst[0], np.asarray(rows_c))
        np.testing.assert_array_equal(dst[1], np.asarray(rows_d))

    def test_split_src_dst_errors_on_unbalanced_split(self) -> None:
        # 1 src block + anchor + 2 dst blocks, but num_layers=2 -> mismatch.
        rows_a = [[0, 1, 2, 3]]
        rows_b = [[10, 11, 12, 13]]
        rows_c = [[20, 21, 22, 23]]
        log_text = "\n".join(
            [
                _mk_tensor_block(rows_a),
                "Loading weights: 100%|#####| 100/100",
                _mk_tensor_block(rows_b),
                _mk_tensor_block(rows_c),
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "fake.log")
            with open(log_path, "w") as fp:
                fp.write(log_text)
            blocks, line_nos, anchor = parse_router_blocks(log_path)
            with self.assertRaisesRegex(AssertionError, "src half"):
                split_src_dst(blocks, line_nos, anchor, num_layers=2)

    def test_dst_to_old_with_reverse_permutation(self) -> None:
        # num_layers=2, num_experts=4, T=1, K=2.
        # perm[L, new] = old. Reverse permutation: perm[L] = [3, 2, 1, 0],
        # so if dst router selects new_id=0, the old id was 3.
        perm = np.asarray([[3, 2, 1, 0], [3, 2, 1, 0]], dtype=np.int64)
        dst_new = np.asarray([[[0, 1]], [[2, 3]]], dtype=np.int64)  # (L=2, T=1, K=2)
        dst_old = dst_to_old(dst_new, perm)
        expected = np.asarray([[[3, 2]], [[1, 0]]], dtype=np.int64)
        np.testing.assert_array_equal(dst_old, expected)

    def test_dst_to_old_rejects_out_of_range(self) -> None:
        perm = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
        dst_new = np.asarray([[[0, 4]]], dtype=np.int64)  # 4 is out of range
        with self.assertRaises(IndexError):
            dst_to_old(dst_new, perm)

    def test_top_k_by_freq_tiebreak_by_id_asc(self) -> None:
        # 5 appears twice, 2/3/7 once each. Tie at count=1 -> id ascending.
        ids = np.asarray([5, 5, 3, 7, 2], dtype=np.int64)
        self.assertEqual(
            top_k_by_freq(ids, k=4),
            [(5, 2), (2, 1), (3, 1), (7, 1)],
        )
        self.assertEqual(
            top_k_by_freq(ids, k=2),
            [(5, 2), (2, 1)],
        )


if __name__ == "__main__":
    unittest.main()
