# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
# nrwu@tencent.com
"""thd.py 单测：PackedSeqParams / make_packed_seq_layout / pack_sequences.

GPU 单进程单卡 bf16 / long；不 load 真权重，秒级跑完。
"""

# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import torch
from torch.utils.checkpoint import checkpoint

from gpatch_v4.models.deepseek_v4.thd import (
    PackedSeqParams,
    _PerMLayout,
    _make_seg_layout,
    cp_slice_layout,
    make_packed_seq_layout,
    pack_sequences,
)


DEVICE = "cuda:0"
SLIDING_WINDOW = 128


def _config(compress_rates: dict[str, int] | None = None) -> SimpleNamespace:
    if compress_rates is None:
        compress_rates = {"compressed_sparse_attention": 4, "heavily_compressed_attention": 128}
    return SimpleNamespace(compress_rates=compress_rates, sliding_window=SLIDING_WINDOW)


def _build_psp(seqlens: list[int], padded_seqlens: list[int]) -> PackedSeqParams:
    cu_q = torch.tensor([0] + list(torch.tensor(seqlens).cumsum(0).tolist()), dtype=torch.int64, device=DEVICE)
    cu_pad = torch.tensor([0] + list(torch.tensor(padded_seqlens).cumsum(0).tolist()), dtype=torch.int64, device=DEVICE)
    return PackedSeqParams(
        cu_seqlens_q=cu_q,
        cu_seqlens_q_padded=cu_pad,
        max_seqlen_q=max(padded_seqlens),
        total_seqlen=sum(padded_seqlens),
    )


class TestPackedSeqParamsBasic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

    def test_psp_basic(self):
        psp = _build_psp([100, 256], [128, 256])
        self.assertEqual(psp.qkv_format, "thd")
        self.assertEqual(psp.max_seqlen_q, 256)
        self.assertEqual(psp.total_seqlen, 384)
        self.assertEqual(psp.cu_seqlens_q.dtype, torch.int64)
        self.assertEqual(psp.cu_seqlens_q.device.type, "cuda")
        self.assertEqual(tuple(psp.cu_seqlens_q.shape), (3,))
        with self.assertRaises(FrozenInstanceError):
            psp.max_seqlen_q = 999  # type: ignore[misc]


class TestPackedSeqParamsPostInit(unittest.TestCase):
    """坏值触发 __post_init__ 的 pure-host check。"""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

    def test_shape_mismatch(self):
        cu_q = torch.tensor([0, 100, 256], dtype=torch.int64, device=DEVICE)
        cu_pad = torch.tensor([0, 128, 256, 512], dtype=torch.int64, device=DEVICE)
        with self.assertRaises(AssertionError):
            PackedSeqParams(
                cu_seqlens_q=cu_q, cu_seqlens_q_padded=cu_pad,
                max_seqlen_q=256, total_seqlen=512,
            )

    def test_dtype_int32(self):
        cu_q = torch.tensor([0, 100, 256], dtype=torch.int32, device=DEVICE)
        cu_pad = torch.tensor([0, 128, 256], dtype=torch.int32, device=DEVICE)
        with self.assertRaises(AssertionError):
            PackedSeqParams(
                cu_seqlens_q=cu_q, cu_seqlens_q_padded=cu_pad,
                max_seqlen_q=256, total_seqlen=256,
            )

    def test_dtype_float(self):
        cu_q = torch.tensor([0.0, 100.0, 256.0], dtype=torch.float32, device=DEVICE)
        cu_pad = torch.tensor([0.0, 128.0, 256.0], dtype=torch.float32, device=DEVICE)
        with self.assertRaises(AssertionError):
            PackedSeqParams(
                cu_seqlens_q=cu_q, cu_seqlens_q_padded=cu_pad,
                max_seqlen_q=256, total_seqlen=256,
            )

    def test_device_mismatch(self):
        cu_q = torch.tensor([0, 100, 256], dtype=torch.int64, device="cpu")
        cu_pad = torch.tensor([0, 128, 256], dtype=torch.int64, device=DEVICE)
        with self.assertRaises(AssertionError):
            PackedSeqParams(
                cu_seqlens_q=cu_q, cu_seqlens_q_padded=cu_pad,
                max_seqlen_q=256, total_seqlen=256,
            )

    def test_qkv_format_bad(self):
        cu_q = torch.tensor([0, 100, 256], dtype=torch.int64, device=DEVICE)
        cu_pad = torch.tensor([0, 128, 256], dtype=torch.int64, device=DEVICE)
        with self.assertRaises(AssertionError):
            PackedSeqParams(
                cu_seqlens_q=cu_q, cu_seqlens_q_padded=cu_pad,
                max_seqlen_q=256, total_seqlen=256, qkv_format="bshd",
            )

    def test_max_seqlen_q_nonpositive(self):
        cu_q = torch.tensor([0, 100, 256], dtype=torch.int64, device=DEVICE)
        cu_pad = torch.tensor([0, 128, 256], dtype=torch.int64, device=DEVICE)
        with self.assertRaises(AssertionError):
            PackedSeqParams(
                cu_seqlens_q=cu_q, cu_seqlens_q_padded=cu_pad,
                max_seqlen_q=0, total_seqlen=256,
            )
        with self.assertRaises(AssertionError):
            PackedSeqParams(
                cu_seqlens_q=cu_q, cu_seqlens_q_padded=cu_pad,
                max_seqlen_q=-1, total_seqlen=256,
            )

    def test_total_seqlen_nonpositive(self):
        cu_q = torch.tensor([0, 100, 256], dtype=torch.int64, device=DEVICE)
        cu_pad = torch.tensor([0, 128, 256], dtype=torch.int64, device=DEVICE)
        with self.assertRaises(AssertionError):
            PackedSeqParams(
                cu_seqlens_q=cu_q, cu_seqlens_q_padded=cu_pad,
                max_seqlen_q=256, total_seqlen=0,
            )


class TestPackedSeqParamsValidate(unittest.TestCase):
    """validate(): 值域 / 单调性 / 一致性（潜在 GPU sync）。"""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

    def _psp(self, cu_q_list, cu_pad_list, max_seqlen_q, total_seqlen=None):
        cu_q = torch.tensor(cu_q_list, dtype=torch.int64, device=DEVICE)
        cu_pad = torch.tensor(cu_pad_list, dtype=torch.int64, device=DEVICE)
        if total_seqlen is None:
            total_seqlen = int(cu_pad[-1])
        return PackedSeqParams(
            cu_seqlens_q=cu_q, cu_seqlens_q_padded=cu_pad,
            max_seqlen_q=max_seqlen_q, total_seqlen=total_seqlen,
        )

    def test_validate_ok(self):
        self._psp([0, 100, 356], [0, 128, 384], 256).validate()

    def test_cu_q_first_nonzero(self):
        with self.assertRaises(AssertionError):
            self._psp([1, 100, 356], [0, 128, 384], 256).validate()

    def test_cu_pad_first_nonzero(self):
        with self.assertRaises(AssertionError):
            self._psp([0, 100, 356], [1, 128, 384], 256).validate()

    def test_cu_q_decreasing(self):
        with self.assertRaises(AssertionError):
            self._psp([0, 200, 100], [0, 256, 256], 256).validate()

    def test_cu_pad_decreasing(self):
        with self.assertRaises(AssertionError):
            self._psp([0, 50, 100], [0, 256, 128], 256).validate()

    def test_q_exceeds_padded_elementwise(self):
        with self.assertRaises(AssertionError):
            self._psp([0, 200, 384], [0, 128, 384], 256).validate()

    def test_max_seqlen_q_inconsistent(self):
        with self.assertRaises(AssertionError):
            self._psp([0, 100, 356], [0, 128, 384], 999).validate()

    def test_total_seqlen_inconsistent(self):
        with self.assertRaises(AssertionError):
            self._psp([0, 100, 356], [0, 128, 384], 256, total_seqlen=999).validate()


class TestCpSliceLayout(unittest.TestCase):
    """cp_slice_layout: per-CP-rank slice mirroring compressor_cp_ring."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

    def test_rank0_no_prefix(self):
        # T=512, cp_size=2, s_local=256; rank 0 has no prefix.
        psp = _build_psp([200, 300], [256, 256])
        layout = make_packed_seq_layout(psp, _config())
        sliced = cp_slice_layout(layout, cp_rank=0, cp_size=2, total_seqlen=512)
        m4 = sliced.per_m[4]
        self.assertEqual(m4.wnd_pos_ids_with_prefix.shape[0], 256 // 4)
        # q-side per-token fields: sliced to s_local (no prefix).
        self.assertEqual(sliced.seg_id_per_token.shape[0], 256)
        self.assertEqual(sliced.pad_token_mask.shape[0], 256)
        self.assertEqual(sliced.seg_id_per_token_with_prefix.shape[0], 256)
        self.assertEqual(m4.causal_threshold_per_token.shape[0], 256)
        # k-side per-window fields stay global.
        self.assertEqual(m4.seg_id_per_wnd.shape[0], 512 // 4)
        # with_prefix per-token / per-window meta.
        self.assertEqual(m4.pad_token_mask_with_prefix.shape[0], 256)
        self.assertEqual(m4.first_of_seg_window_mask_with_prefix.shape[0], 256 // 4)

    def test_rank_gt0_with_prefix(self):
        # T=512, cp_size=2, s_local=256; rank 1 prepends m tokens from prev rank.
        psp = _build_psp([200, 300], [256, 256])
        layout = make_packed_seq_layout(psp, _config())
        sliced = cp_slice_layout(layout, cp_rank=1, cp_size=2, total_seqlen=512)
        m4 = sliced.per_m[4]
        # wnd = pos_ids[::4] over 260 tokens = 65 windows.
        self.assertEqual(m4.wnd_pos_ids_with_prefix.shape[0], (4 + 256) // 4)
        # q-side per-token fields: sliced to s_local (no prefix).
        self.assertEqual(sliced.seg_id_per_token.shape[0], 256)
        self.assertEqual(m4.causal_threshold_per_token.shape[0], 256)
        self.assertEqual(
            sliced.seg_id_per_token_with_prefix.shape[0],
            256 + (SLIDING_WINDOW - 1),
        )
        # k-side per-window fields stay global.
        self.assertEqual(m4.seg_id_per_wnd.shape[0], 512 // 4)
        # with_prefix per-token / per-window meta: include the m-token prefix.
        self.assertEqual(m4.pad_token_mask_with_prefix.shape[0], 4 + 256)
        self.assertEqual(m4.first_of_seg_window_mask_with_prefix.shape[0], (4 + 256) // 4)

    def test_per_query_wnd_fields_values(self):
        """q-side fields sliced to s_local; values still in GLOBAL window-index space.

        Setup: seqlens=[200, 300], padded=[256, 256], T=512.
        rank 0 owns tokens 0..255 (all in seg 0):
            causal_threshold[t] = (t + 1) // 4
        rank 1 owns tokens 256..511 (all in seg 1):
            causal_threshold[t_local] = 64 + (t_local + 1) // 4
        """
        psp = _build_psp([200, 300], [256, 256])
        layout = make_packed_seq_layout(psp, _config())

        sliced0 = cp_slice_layout(layout, cp_rank=0, cp_size=2, total_seqlen=512)
        m4_0 = sliced0.per_m[4]
        self.assertEqual(int(m4_0.causal_threshold_per_token[0]), 0)
        self.assertEqual(int(m4_0.causal_threshold_per_token[3]), 1)
        self.assertEqual(int(m4_0.causal_threshold_per_token[255]), 64)

        sliced1 = cp_slice_layout(layout, cp_rank=1, cp_size=2, total_seqlen=512)
        m4_1 = sliced1.per_m[4]
        self.assertEqual(int(m4_1.causal_threshold_per_token[0]), 64)
        self.assertEqual(int(m4_1.causal_threshold_per_token[3]), 65)
        self.assertEqual(int(m4_1.causal_threshold_per_token[255]), 128)

    def test_m_keys_preserved(self):
        psp = _build_psp([100, 200], [128, 384])
        layout = make_packed_seq_layout(psp, _config())
        sliced = cp_slice_layout(layout, cp_rank=1, cp_size=2, total_seqlen=512)
        self.assertEqual(set(sliced.per_m.keys()), {4, 128})

    def test_returns_new_object(self):
        psp = _build_psp([200, 300], [256, 256])
        layout = make_packed_seq_layout(psp, _config())
        original_pos = layout.per_m[4].wnd_pos_ids_with_prefix.clone()
        sliced = cp_slice_layout(layout, cp_rank=1, cp_size=2, total_seqlen=512)
        # original layout untouched (the function is non-destructive).
        self.assertTrue(torch.equal(layout.per_m[4].wnd_pos_ids_with_prefix, original_pos))
        self.assertIsNot(sliced, layout)
        self.assertIsNot(sliced.per_m[4], layout.per_m[4])

    def test_with_prefix_meta_values(self):
        """pad_token_mask_with_prefix / first_of_seg_window_mask_with_prefix
        carry the per-rank-with-prefix view of pad / seg-boundary flags.

        Setup: seqlens=[200, 100], padded=[256, 256], T=512.
        Pad slots: seg 0 tokens 200..255 (56 pad), seg 1 tokens 356..511 (156 pad)
            → global pad_token_mask: [F]*200 + [T]*56 + [F]*100 + [T]*156
        m=4 first_of_seg_window_mask_with_prefix (before cp_slice):
            → True only at idx 64 (seg 1 start).
        """
        psp = _build_psp([200, 100], [256, 256])
        layout = make_packed_seq_layout(psp, _config())

        # Rank 0 (no prefix): tokens 0..255, windows 0..63 (m=4).
        s0 = cp_slice_layout(layout, cp_rank=0, cp_size=2, total_seqlen=512)
        m4_0 = s0.per_m[4]
        # rank 0 pad: F*200 + T*56
        expected_pad0 = torch.cat([
            torch.zeros(200, dtype=torch.bool, device=DEVICE),
            torch.ones(56, dtype=torch.bool, device=DEVICE),
        ])
        self.assertTrue(torch.equal(m4_0.pad_token_mask_with_prefix, expected_pad0))
        # rank 0 first_of_seg covers wnds 0..63 (cu_n_wnd_padded=[0,64,128]) → all False.
        self.assertFalse(m4_0.first_of_seg_window_mask_with_prefix.any().item())

        # Rank 1 (m=4 prefix): tokens 252..511 (260 tokens), windows 63..127 (65 windows).
        s1 = cp_slice_layout(layout, cp_rank=1, cp_size=2, total_seqlen=512)
        m4_1 = s1.per_m[4]
        # rank 1 pad over tokens 252..511: prefix tokens 252..255 (4 pad, all in seg 0 pad
        # tail) + seg 1 tokens 256..355 (100 effective) + 356..511 (156 pad).
        expected_pad1 = torch.cat([
            torch.ones(4, dtype=torch.bool, device=DEVICE),
            torch.zeros(100, dtype=torch.bool, device=DEVICE),
            torch.ones(156, dtype=torch.bool, device=DEVICE),
        ])
        self.assertTrue(torch.equal(m4_1.pad_token_mask_with_prefix, expected_pad1))
        # rank 1 first_of_seg over wnds 63..127: True only at wnd 64
        # (seg 1 start), which is the second window of this view (index 1).
        expected_first1 = torch.zeros(65, dtype=torch.bool, device=DEVICE)
        expected_first1[1] = True
        self.assertTrue(torch.equal(m4_1.first_of_seg_window_mask_with_prefix, expected_first1))

    def test_seg_id_per_token_full_stays_global(self):
        """``seg_id_per_token_full`` is **not** sliced by cp_slice_layout.

        ``seg_id_per_token_with_prefix`` carries the SWA kv-buffer view
        (``l_swa_prefix = sliding_window - 1`` on rank > 0).
        """
        psp = _build_psp([200, 300], [256, 256])
        layout = make_packed_seq_layout(psp, _config())
        # 出厂时 _full == seg_id_per_token，长度 = T。
        self.assertEqual(layout.seg_id_per_token_full.shape[0], 512)
        self.assertTrue(torch.equal(
            layout.seg_id_per_token_full, layout.seg_id_per_token,
        ))
        self.assertTrue(torch.equal(
            layout.seg_id_per_token_with_prefix, layout.seg_id_per_token,
        ))

        # cp_slice_layout 后：seg_id_per_token 切到 [s_local]，但 _full 不切。
        for cp_rank in (0, 1):
            sliced = cp_slice_layout(
                layout, cp_rank=cp_rank, cp_size=2, total_seqlen=512,
            )
            self.assertEqual(sliced.seg_id_per_token.shape[0], 256)
            self.assertEqual(sliced.seg_id_per_token_full.shape[0], 512)
            self.assertTrue(torch.equal(
                sliced.seg_id_per_token_full, layout.seg_id_per_token_full,
            ))
            # 切片后的 seg_id_per_token 应等于 _full 切对应区间。
            start = cp_rank * 256
            self.assertTrue(torch.equal(
                sliced.seg_id_per_token,
                layout.seg_id_per_token_full[start : start + 256],
            ))
            l_swa_prefix = 0 if cp_rank == 0 else SLIDING_WINDOW - 1
            self.assertEqual(sliced.seg_id_per_token_with_prefix.shape[0], 256 + l_swa_prefix)
            self.assertTrue(torch.equal(
                sliced.seg_id_per_token_with_prefix,
                layout.seg_id_per_token_full[start - l_swa_prefix : start + 256],
            ))


class TestPackSequences(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

    def _seq(self, length, start_id):
        return torch.arange(start_id, start_id + length, dtype=torch.long, device=DEVICE)

    def test_lengths(self):
        seqs = [self._seq(s, s * 1000) for s in [64, 100, 128, 256, 33]]
        ids, pos, labs, psp = pack_sequences(seqs, None, config=_config(), pad_to_multiple_of=128)
        self.assertEqual(psp.cu_seqlens_q.tolist(), [0, 64, 164, 292, 548, 581])
        self.assertEqual(psp.cu_seqlens_q_padded.tolist(), [0, 128, 256, 384, 640, 768])
        self.assertEqual(psp.max_seqlen_q, 256)
        self.assertEqual(psp.total_seqlen, 768)
        self.assertEqual(ids.shape, (1, 768))
        self.assertEqual(pos.shape, (1, 768))
        self.assertIsNone(labs)
        self.assertIsNotNone(psp.layout)

    def test_position_ids(self):
        seqs = [self._seq(s, 0) for s in [3, 5]]
        # pad=4 → padded lens [4, 8]; need a config whose every m divides every padded len.
        _, pos, _, _ = pack_sequences(
            seqs, None, config=_config({"compressed_sparse_attention": 4}), pad_to_multiple_of=4,
        )
        # padded seqlens: 4, 8 → T = 12; seg-local pos: [0,1,2,3, 0,1,2,3,4,5,6,7]
        expected = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long, device=DEVICE)
        self.assertTrue(torch.equal(pos.squeeze(0), expected))

    def test_labels_mask(self):
        seqs = [self._seq(s, 0) for s in [3, 5]]
        labs_in = [torch.full((3,), 7, dtype=torch.long, device=DEVICE),
                   torch.full((5,), 9, dtype=torch.long, device=DEVICE)]
        _, _, labs, _ = pack_sequences(
            seqs, labs_in, config=_config({"compressed_sparse_attention": 4}), pad_to_multiple_of=4,
        )
        # seg 0 (padded len 4): real [7,7,7] then pad [-100]; seg-final = -100
        # → [7, 7, -100, -100]
        # seg 1 (padded len 8): real [9,9,9,9,9] then pad [-100]*3; seg-final = -100
        # → [9, 9, 9, 9, -100, -100, -100, -100]
        expected = torch.tensor(
            [7, 7, -100, -100, 9, 9, 9, 9, -100, -100, -100, -100],
            dtype=torch.long, device=DEVICE,
        )
        assert labs is not None
        self.assertTrue(torch.equal(labs.squeeze(0), expected))

    def test_no_labels(self):
        seqs = [self._seq(s, 0) for s in [10, 20]]
        ids, _, labs, psp = pack_sequences(seqs, None, config=_config(), pad_to_multiple_of=128)
        self.assertIsNone(labs)
        self.assertEqual(ids.shape, (1, 256))
        self.assertEqual(psp.max_seqlen_q, 128)

    def test_already_multiple(self):
        seqs = [self._seq(128, 0), self._seq(256, 0)]
        ids, _, _, psp = pack_sequences(seqs, None, config=_config(), pad_to_multiple_of=128)
        # 已是整数倍 → 不重复 pad
        self.assertEqual(psp.cu_seqlens_q.tolist(), [0, 128, 384])
        self.assertEqual(psp.cu_seqlens_q_padded.tolist(), [0, 128, 384])
        self.assertEqual(ids.shape[1], 384)

    def test_edge_cases(self):
        # 空 list
        with self.assertRaises(ValueError):
            pack_sequences([], None, config=_config(), pad_to_multiple_of=128)
        # s_i == 0
        with self.assertRaises(ValueError):
            pack_sequences(
                [torch.empty(0, dtype=torch.long, device=DEVICE)],
                None, config=_config(), pad_to_multiple_of=128,
            )
        # 单段 OK
        ids, _, _, psp = pack_sequences(
            [self._seq(50, 0)], None, config=_config(), pad_to_multiple_of=128,
        )
        self.assertEqual(psp.cu_seqlens_q_padded.tolist(), [0, 128])
        self.assertEqual(ids.shape, (1, 128))
        # labels 与 input_ids 长度不一致
        with self.assertRaises(ValueError):
            pack_sequences(
                [self._seq(10, 0)],
                [torch.empty(99, dtype=torch.long, device=DEVICE)],
                config=_config(),
                pad_to_multiple_of=128,
            )
        # input_ids 非 1D
        with self.assertRaises(ValueError):
            pack_sequences(
                [torch.empty(2, 3, dtype=torch.long, device=DEVICE)],
                None,
                config=_config(),
                pad_to_multiple_of=128,
            )
        # labels[i] shape 不匹配
        with self.assertRaises(ValueError):
            pack_sequences(
                [self._seq(10, 0)],
                [torch.empty(11, dtype=torch.long, device=DEVICE)],
                config=_config(),
                pad_to_multiple_of=128,
            )
        # pad_to_multiple_of <= 0
        with self.assertRaises(ValueError):
            pack_sequences([self._seq(10, 0)], None, config=_config(), pad_to_multiple_of=0)
        # cross-device input_ids
        with self.assertRaises(ValueError):
            pack_sequences(
                [self._seq(10, 0), torch.zeros(5, dtype=torch.long, device="cpu")],
                None,
                config=_config(),
                pad_to_multiple_of=128,
            )

    def test_dtypes_devices(self):
        seqs = [self._seq(10, 0), self._seq(20, 0)]
        labs = [torch.full((10,), 1, dtype=torch.long, device=DEVICE),
                torch.full((20,), 2, dtype=torch.long, device=DEVICE)]
        ids, pos, l_out, psp = pack_sequences(
            seqs, labs, config=_config(), pad_to_multiple_of=128,
        )
        self.assertEqual(ids.dtype, torch.long)
        self.assertEqual(pos.dtype, torch.int64)
        assert l_out is not None
        self.assertEqual(l_out.dtype, torch.long)
        self.assertEqual(psp.cu_seqlens_q.dtype, torch.int64)
        self.assertEqual(psp.cu_seqlens_q_padded.dtype, torch.int64)
        for t in [ids, pos, l_out, psp.cu_seqlens_q, psp.cu_seqlens_q_padded]:
            self.assertEqual(t.device.type, "cuda")


class TestMakePackedSeqLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

    def _setup(self, seqlens, padded_seqlens):
        psp = _build_psp(seqlens, padded_seqlens)
        T = sum(padded_seqlens)
        return psp, T

    def test_seg_three(self):
        psp, T = self._setup([10, 20, 30], [128, 128, 128])
        layout = make_packed_seq_layout(psp, _config())
        # seg_id_per_token: 0 for [0..127], 1 for [128..255], 2 for [256..383]
        expected_seg_id = torch.cat([
            torch.full((128,), i, dtype=torch.long, device=DEVICE) for i in range(3)
        ])
        self.assertTrue(torch.equal(layout.seg_id_per_token, expected_seg_id))
        # pad_token_mask: True at trailing s_padded - s slots of each seg
        pad_per_seg = []
        for s, s_p in zip([10, 20, 30], [128, 128, 128]):
            mask = torch.ones(s_p, dtype=torch.bool, device=DEVICE)
            mask[:s] = False
            pad_per_seg.append(mask)
        expected_pad = torch.cat(pad_per_seg)
        self.assertTrue(torch.equal(layout.pad_token_mask, expected_pad))
        # seg_id_per_wnd lives on each _PerMLayout (length = T // m).
        # m=128 (HCA): 1 wnd per seg → [0, 1, 2]
        self.assertTrue(torch.equal(
            layout.per_m[128].seg_id_per_wnd,
            torch.tensor([0, 1, 2], dtype=torch.long, device=DEVICE),
        ))
        # m=4 (CSA): 32 wnds per seg → [0]*32 + [1]*32 + [2]*32
        expected_m4 = torch.cat([
            torch.full((32,), i, dtype=torch.long, device=DEVICE) for i in range(3)
        ])
        self.assertTrue(torch.equal(layout.per_m[4].seg_id_per_wnd, expected_m4))

    def test_seg_id_per_wnd_uneven(self):
        # padded_seqlens [128, 256, 128] → m=128 wnds = [0, 1, 1, 2]
        psp, T = self._setup([100, 200, 80], [128, 256, 128])
        layout = make_packed_seq_layout(psp, _config())
        self.assertTrue(torch.equal(
            layout.per_m[128].seg_id_per_wnd,
            torch.tensor([0, 1, 1, 2], dtype=torch.long, device=DEVICE),
        ))
        # m=4: per-seg windows = [32, 64, 32]
        expected_m4 = torch.cat([
            torch.full((32,), 0, dtype=torch.long, device=DEVICE),
            torch.full((64,), 1, dtype=torch.long, device=DEVICE),
            torch.full((32,), 2, dtype=torch.long, device=DEVICE),
        ])
        self.assertTrue(torch.equal(layout.per_m[4].seg_id_per_wnd, expected_m4))

    def test_per_m_keys(self):
        psp, T = self._setup([10, 20], [128, 128])
        layout = make_packed_seq_layout(psp, _config())
        self.assertEqual(set(layout.per_m.keys()), {4, 128})
        # 重复值 → fail-fast
        bad_config = _config({"a": 4, "b": 4})
        with self.assertRaises(AssertionError):
            make_packed_seq_layout(psp, bad_config)

    def test_per_m_causal_threshold(self):
        # padded_seqlens [128, 256, 128]
        # causal_threshold_per_token = (global_pos + 1) // 4
        # seg 0 token 0 (global_pos=0): 0
        # seg 0 token 7 (global_pos=7): 2
        # seg 1 token 0 (global_pos=128): 32
        # seg 1 token 11 (global_pos=139): 35
        psp, T = self._setup([100, 200, 80], [128, 256, 128])
        layout = make_packed_seq_layout(psp, _config())
        m4 = layout.per_m[4]
        self.assertEqual(int(m4.causal_threshold_per_token[0]), 0)
        self.assertEqual(int(m4.causal_threshold_per_token[7]), 2)
        self.assertEqual(int(m4.causal_threshold_per_token[128]), 32)
        self.assertEqual(int(m4.causal_threshold_per_token[139]), 35)

    def test_first_of_seg(self):
        psp, T = self._setup([100, 200, 80], [128, 256, 128])
        layout = make_packed_seq_layout(psp, _config())
        m4 = layout.per_m[4]
        # first_of_seg_window_mask_with_prefix True at windows 32 and 96 only (skip 0 and trailing 128).
        expected = torch.zeros(128, dtype=torch.bool, device=DEVICE)
        expected[32] = True
        expected[96] = True
        self.assertTrue(torch.equal(m4.first_of_seg_window_mask_with_prefix, expected))
        m128 = layout.per_m[128]
        # first_of_seg True at 1 and 3.
        expected128 = torch.tensor([False, True, False, True], device=DEVICE)
        self.assertTrue(torch.equal(m128.first_of_seg_window_mask_with_prefix, expected128))

    def test_first_of_seg_single_segment(self):
        psp, T = self._setup([100], [128])
        layout = make_packed_seq_layout(psp, _config())
        # N=1 退化：mask 应全 False（没有非 0 段起点要打 True）
        self.assertFalse(layout.per_m[4].first_of_seg_window_mask_with_prefix.any().item())
        self.assertFalse(layout.per_m[128].first_of_seg_window_mask_with_prefix.any().item())

    def test_per_m_frozen(self):
        psp, T = self._setup([10], [128])
        layout = make_packed_seq_layout(psp, _config())
        m4 = layout.per_m[4]
        with self.assertRaises(FrozenInstanceError):
            m4.max_n_wnd = 999  # type: ignore[misc]
        # outer container is also frozen
        with self.assertRaises(FrozenInstanceError):
            layout.seg_id_per_token = torch.zeros(T, dtype=torch.long, device=DEVICE)  # type: ignore[misc]


class _LayoutConsumer(torch.nn.Module):
    """Minimal nn.Module that consumes layout fields + a learnable param.

    Used for recompute safety tests: input goes through a linear, then we
    multiply by a layout-derived float mask and sum. Backward thus requires
    the linear's weight to receive grad regardless of recompute settings.
    """

    def __init__(self, hidden):
        super().__init__()
        self.lin = torch.nn.Linear(hidden, hidden, bias=False)

    def forward(self, x, layout):
        # layout fields are ints/bools — promote to float for the multiplicative path.
        seg_w = layout.seg_id_per_token.to(x.dtype)
        pad_w = layout.pad_token_mask.to(x.dtype)
        m4_w = layout.per_m[4].causal_threshold_per_token.to(x.dtype)
        h = self.lin(x)
        return (h * seg_w.unsqueeze(-1) * pad_w.unsqueeze(-1) * m4_w.unsqueeze(-1)).sum()


class TestRecomputeSafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

    def _setup(self):
        psp = _build_psp([100, 200], [128, 256])
        T = 384
        layout = make_packed_seq_layout(psp, _config())
        return layout, T

    def _run(self, use_reentrant: bool):
        layout, T = self._setup()
        torch.manual_seed(0)
        hidden = 8
        x = torch.randn(T, hidden, device=DEVICE, dtype=torch.float32, requires_grad=True)

        torch.manual_seed(42)
        m_base = _LayoutConsumer(hidden).to(DEVICE)
        torch.manual_seed(42)
        m_ckpt = _LayoutConsumer(hidden).to(DEVICE)

        # baseline: no checkpoint
        loss_base = m_base(x, layout)
        loss_base.backward()
        grad_base_x = x.grad.detach().clone()
        grad_base_w = m_base.lin.weight.grad.detach().clone()

        x.grad = None
        # ckpt: same module weights, ckpt path
        loss_ckpt = checkpoint(m_ckpt, x, layout, use_reentrant=use_reentrant)
        loss_ckpt.backward()
        grad_ckpt_x = x.grad.detach().clone()
        grad_ckpt_w = m_ckpt.lin.weight.grad.detach().clone()

        self.assertTrue(torch.equal(loss_base, loss_ckpt))
        self.assertTrue(torch.equal(grad_base_x, grad_ckpt_x))
        self.assertTrue(torch.equal(grad_base_w, grad_ckpt_w))

    def test_reentrant(self):
        self._run(use_reentrant=True)

    def test_non_reentrant(self):
        self._run(use_reentrant=False)


class _LayoutMutator(torch.nn.Module):
    """Adversarial: in-place mutates layout mid-forward (violates contract)."""

    def __init__(self, hidden):
        super().__init__()
        self.lin = torch.nn.Linear(hidden, hidden, bias=False)

    def forward(self, x, layout):
        h = self.lin(x)
        m4_w = layout.per_m[4].causal_threshold_per_token.to(x.dtype)
        out = (h * m4_w.unsqueeze(-1)).sum()
        # CONTRACT VIOLATION: in-place mutate layout after consumption. Under
        # reentrant ckpt this changes recompute's view.
        layout.per_m[4].causal_threshold_per_token.add_(1000)
        return out


class TestRecomputeMutateNegative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

    def test_frozen_setattr(self):
        psp = _build_psp([10], [128])
        T = 128
        layout = make_packed_seq_layout(psp, _config())
        with self.assertRaises(FrozenInstanceError):
            layout.per_m[4].max_n_wnd = 5  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            layout.per_m[4].cu_n_wnd_padded = torch.zeros(  # type: ignore[misc]
                1, device=DEVICE, dtype=torch.long,
            )

    def test_mutate_breaks_reentrant_recompute(self):
        psp = _build_psp([100, 200], [128, 256])
        T = 384

        layout_a = make_packed_seq_layout(psp, _config())
        layout_b = make_packed_seq_layout(psp, _config())

        torch.manual_seed(0)
        x = torch.randn(T, 8, device=DEVICE, dtype=torch.float32, requires_grad=True)

        torch.manual_seed(7)
        m_base = _LayoutConsumer(8).to(DEVICE)
        torch.manual_seed(7)
        m_mut = _LayoutMutator(8).to(DEVICE)

        loss_clean = m_base(x, layout_a)
        loss_clean.backward()
        grad_clean_x = x.grad.detach().clone()

        x.grad = None
        loss_dirty = checkpoint(m_mut, x, layout_b, use_reentrant=True)
        loss_dirty.backward()
        grad_dirty_x = x.grad.detach().clone()

        # 反例期望：mutate 后 reentrant recompute 看到的 layout 已变，grad 与 clean 必定不一致。
        self.assertFalse(torch.equal(grad_clean_x, grad_dirty_x))


class TestMakeSegMetaPrivate(unittest.TestCase):
    """Sanity for the private helper, called by both layout and tests."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

    def test_basic(self):
        # cu_q=[0,5,12]: seg 0 has 5 effective tokens, seg 1 has 7 (=12-5).
        # cu_pad=[0,8,16]: layout is [seg0 pad to 8 | seg1 pad to 8] = 16 total.
        cu_q = torch.tensor([0, 5, 12], dtype=torch.int64, device=DEVICE)
        cu_pad = torch.tensor([0, 8, 16], dtype=torch.int64, device=DEVICE)
        seg_id, pad = _make_seg_layout(cu_q, cu_pad, 16)
        # Token positions 0..7 → seg 0; 8..15 → seg 1
        expected_seg = torch.tensor([0]*8 + [1]*8, dtype=torch.long, device=DEVICE)
        self.assertTrue(torch.equal(seg_id, expected_seg))
        # seg 0 pad at last 3 (positions 5..7); seg 1 pad at last 1 (position 15).
        expected_pad = torch.tensor(
            [False]*5 + [True]*3 + [False]*7 + [True]*1,
            device=DEVICE,
        )
        self.assertTrue(torch.equal(pad, expected_pad))


if __name__ == "__main__":
    unittest.main()
