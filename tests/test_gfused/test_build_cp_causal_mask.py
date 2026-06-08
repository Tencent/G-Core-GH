# coding=utf-8
# copyright (c) 2026 tencent inc. all rights reserved.
# nrwu@tencent.com
"""Unit tests for :func:`build_cp_causal_mask`.

Pure CPU tests — no Ray cluster or GPU needed. Run with::

    cd /work/wepsdl/gcore-dev
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s tests/test_gfused/test_build_cp_causal_mask.py
"""

import unittest

import torch

from gpatch_v4.models.deepseek_v4.cp import build_cp_causal_mask


class TestBuildCpCausalMask(unittest.TestCase):
    """Verify :func:`build_cp_causal_mask` shape, values, and semantics.

    Pure CPU tests — no Ray cluster or GPU needed.
    """

    @staticmethod
    def _display_mask(
        mask: torch.Tensor,
        s_local: int,
        cp_rank: int,
        swa_prefix_len: int,
        sliding_window: int,
    ) -> None:
        """Pretty-print the mask grid for human inspection.

        ``■`` = visible (0.0), ``·`` = blocked (-inf).
        Row/column headers show absolute token positions.
        """
        m = mask.squeeze(0).squeeze(0)  # [s_local, s_local + swa_prefix_len]
        q_start = cp_rank * s_local
        k_start = cp_rank * s_local - swa_prefix_len
        n_cols = m.shape[1]

        print(
            f"\n{'=' * 60}\n"
            f"build_cp_causal_mask(s_local={s_local}, cp_rank={cp_rank}, "
            f"swa_prefix_len={swa_prefix_len}, sliding_window={sliding_window})\n"
            f"  shape       = {list(mask.shape)}\n"
            f"  Q abs range = [{q_start}, {q_start + s_local})\n"
            f"  K abs range = [{k_start}, {k_start + n_cols})"
        )

        cw = 4  # column width
        # Ring / local separator position in columns
        sep_col = swa_prefix_len  # columns [0, swa_prefix_len) are ring prefix

        # Column header (absolute K positions)
        hdr = " " * (cw + 1)
        for j in range(n_cols):
            hdr += f"{k_start + j:>{cw}}"
        print(hdr)

        # Separator between ring prefix and local KV in header
        if swa_prefix_len > 0:
            sep = " " * (cw + 1)
            for j in range(n_cols):
                if j == sep_col:
                    sep += f"{'|':>{cw}}"
                else:
                    sep += " " * cw
            print(sep)

        for i in range(s_local):
            q_abs = q_start + i
            row = f"{q_abs:>{cw}} "
            for j in range(n_cols):
                ch = "■" if m[i, j].item() == 0.0 else "·"
                row += f"{ch:>{cw}}"
            print(row)

    def test_shape_and_dtype(self):
        """Output shape is [1, 1, s_local, s_local + swa_prefix_len]; values are 0 or -inf."""

        for s_local, cp_rank, swa_prefix_len, sw in [
            (8, 0, 0, 4),
            (8, 1, 3, 4),
            (8, 2, 3, 4),
            (16, 0, 0, 8),
            (16, 3, 7, 8),
        ]:
            mask = build_cp_causal_mask(
                s_local, cp_rank, swa_prefix_len, sw, device=torch.device("cpu"),
            )
            self.assertEqual(mask.shape, (1, 1, s_local, s_local + swa_prefix_len))
            self.assertEqual(mask.dtype, torch.float32)
            vals = mask.unique()
            for v in vals:
                self.assertTrue(
                    v.item() == 0.0 or v.item() == float("-inf"),
                    f"unexpected value {v.item()}",
                )

    def test_causal_property(self):
        """No query can attend to a future KV token (q_abs < k_abs → -inf)."""

        for s_local, cp_rank, swa_prefix_len, sw in [
            (8, 0, 0, 4),
            (8, 1, 3, 4),
            (8, 2, 3, 4),
        ]:
            mask = build_cp_causal_mask(
                s_local, cp_rank, swa_prefix_len, sw, device=torch.device("cpu"),
            )
            m = mask.squeeze(0).squeeze(0)
            q_start = cp_rank * s_local
            k_start = cp_rank * s_local - swa_prefix_len

            for i in range(s_local):
                for j in range(s_local + swa_prefix_len):
                    q_abs, k_abs = q_start + i, k_start + j
                    if q_abs < k_abs:
                        self.assertEqual(
                            m[i, j].item(), float("-inf"),
                            f"cp_rank={cp_rank}: q_abs={q_abs} < k_abs={k_abs} "
                            f"should be masked, got 0.0",
                        )

    def test_sliding_window_property(self):
        """Tokens outside the sliding window (q - k >= sliding_window) are blocked."""

        for s_local, cp_rank, swa_prefix_len, sw in [
            (8, 0, 0, 4),
            (8, 1, 3, 4),
            (16, 3, 7, 8),
        ]:
            mask = build_cp_causal_mask(
                s_local, cp_rank, swa_prefix_len, sw, device=torch.device("cpu"),
            )
            m = mask.squeeze(0).squeeze(0)
            q_start = cp_rank * s_local
            k_start = cp_rank * s_local - swa_prefix_len

            for i in range(s_local):
                for j in range(s_local + swa_prefix_len):
                    q_abs, k_abs = q_start + i, k_start + j
                    if q_abs - k_abs >= sw:
                        self.assertEqual(
                            m[i, j].item(), float("-inf"),
                            f"cp_rank={cp_rank}: q_abs={q_abs}, k_abs={k_abs}, "
                            f"dist={q_abs - k_abs} >= sw={sw} should be masked",
                        )

    def test_visible_region_exact(self):
        """Visible iff ``q >= k`` AND ``q - k < sliding_window``."""

        for s_local, cp_rank, swa_prefix_len, sw in [
            (8, 0, 0, 4),
            (8, 1, 3, 4),
            (8, 2, 3, 4),
            (16, 0, 0, 8),
            (16, 3, 7, 8),
        ]:
            mask = build_cp_causal_mask(
                s_local, cp_rank, swa_prefix_len, sw, device=torch.device("cpu"),
            )
            m = mask.squeeze(0).squeeze(0)
            q_start = cp_rank * s_local
            k_start = cp_rank * s_local - swa_prefix_len

            for i in range(s_local):
                for j in range(s_local + swa_prefix_len):
                    q_abs, k_abs = q_start + i, k_start + j
                    should_see = (q_abs >= k_abs) and (q_abs - k_abs < sw)
                    actual = m[i, j].item() == 0.0
                    self.assertEqual(
                        actual, should_see,
                        f"cp_rank={cp_rank}, q={q_abs}, k={k_abs}: "
                        f"expected visible={should_see}, got {actual}",
                    )

    def test_rank0_matches_standard_causal_sw(self):
        """Rank 0 (swa_prefix_len=0) should equal a standard causal + sliding-window mask."""

        s_local, sw = 16, 6
        mask = build_cp_causal_mask(
            s_local, cp_rank=0, swa_prefix_len=0, sliding_window=sw,
            device=torch.device("cpu"),
        )
        m = mask.squeeze(0).squeeze(0)

        # Reference: standard lower-triangular banded mask
        ref = torch.zeros(s_local, s_local)
        for i in range(s_local):
            for j in range(s_local):
                if not (i >= j and i - j < sw):
                    ref[i, j] = float("-inf")
        self.assertTrue(torch.equal(m, ref))

    def test_cross_rank_continuity(self):
        """Across all CP ranks, the union of visible regions equals the
        full-sequence causal + sliding-window mask (each q-row sees the same
        set of absolute K positions as the non-CP version).
        """

        s_local, cp_size, sw = 8, 4, 6
        s_full = s_local * cp_size

        ref = torch.zeros(s_full, s_full)
        for i in range(s_full):
            for j in range(s_full):
                if not (i >= j and i - j < sw):
                    ref[i, j] = float("-inf")

        for cp_rank in range(cp_size):
            swa_prefix_len = min(sw - 1, cp_rank * s_local)
            mask = build_cp_causal_mask(
                s_local, cp_rank, swa_prefix_len, sw, device=torch.device("cpu"),
            )
            m = mask.squeeze(0).squeeze(0)
            q_start = cp_rank * s_local
            k_start = cp_rank * s_local - swa_prefix_len

            for i in range(s_local):
                for j in range(s_local + swa_prefix_len):
                    q_abs, k_abs = q_start + i, k_start + j
                    expected = ref[q_abs, k_abs].item()
                    actual = m[i, j].item()
                    self.assertEqual(
                        actual, expected,
                        f"cp_rank={cp_rank}, q_abs={q_abs}, k_abs={k_abs}: "
                        f"mask={actual} != ref={expected}",
                    )

    def test_display(self):
        """Visual display of masks for human review — always passes."""

        s_local, sw, cp_size = 8, 4, 4

        print("\n\n" + "=" * 60)
        print("Visual display of build_cp_causal_mask")
        print(f"s_local={s_local}, sliding_window={sw}, cp_size={cp_size}")
        print(f"S_total = {s_local * cp_size}")
        print("■ = visible (0.0)    · = blocked (-inf)")
        print(f"Column '|' separates ring-prefix KV from local KV")

        for cp_rank in range(cp_size):
            swa_prefix_len = min(sw - 1, cp_rank * s_local)
            mask = build_cp_causal_mask(
                s_local, cp_rank, swa_prefix_len, sw, device=torch.device("cpu"),
            )
            self._display_mask(mask, s_local, cp_rank, swa_prefix_len, sw)
