# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
# nrwu@tencent.com
"""DeepSeek-V4 TileLang kernel op 单元测试（从 miles 移植后的 cleanup 验证）。

覆盖两个 op：
  * sparse_attn_tilelang  —— 稀疏 MQA attention（fwd + bwd），对照 fp32 PyTorch 参考实现
  * v4_lighting_indexer / batched_indexer_fwd —— Lightning Indexer 打分 + top-k

判据：fused kernel（bf16 计算 + fp32 累加）与纯 fp32 参考在 bf16 容差内一致。
另覆盖 dynamic topk：边界 pad、重 mask、-1 pad 等价性、多 topk 共用同一 JIT specialization。
单进程单卡，需要 GPU（tilelang JIT 编译真实 CUDA kernel）。

Usage::

    cd /work/wepsdl/gcore-dev
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$PYTHONPATH"
    pytest -v -s --timeout=1800 tests/test_gfused/test_deepseek_v4_kernel_ops.py
"""

import unittest

import torch
import torch.nn.functional as F

from gpatch_v4.models.deepseek_v4.kernel.tilelang_indexer import v4_lighting_indexer
from gpatch_v4.models.deepseek_v4.kernel.tilelang_indexer_fwd import (
    _make_causal_cu_seqlens,
    batched_indexer_fwd,
)
from gpatch_v4.models.deepseek_v4.kernel.tilelang_sparse_mla_bwd import bwd as sparse_mqa_bwd
from gpatch_v4.models.deepseek_v4.kernel.tilelang_sparse_mla_fwd import sparse_mqa_fwd
from gpatch_v4.models.deepseek_v4.kernel.tilelang_sparse_mla import sparse_attn_tilelang

DEVICE = "cuda:0"


def _worst_diff(actual: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """返回 (max_abs_diff, max_rel_diff)，rel 以 ref 的最大绝对值为分母。"""
    a = actual.detach().float()
    r = ref.detach().float()
    abs_d = (a - r).abs().max().item()
    rel_d = abs_d / max(r.abs().max().item(), 1e-12)
    return abs_d, rel_d


_SPARSE_ATTN_CASES = [
    # (H, D, S, S_KV, TOPK, label)
    # --- 原 miles 参数 ---
    (512, 256, 128, 160, 132, "miles_S128"),
    (512, 256, 1024, 1280, 132, "miles_S1024"),
    (512, 256, 4096, 5120, 132, "miles_S4096"),
    # --- DSV4-Flash 真实参数 (H=64, D=512) ---
    (64, 512, 128, 128, 128, "dsv4_L0_cp0"),       # sliding-only, cp_rank=0
    (64, 512, 128, 255, 128, "dsv4_L0_cp1"),        # sliding-only, cp_rank>0 (128+127 prefix)
    (64, 512, 128, 192, 192, "dsv4_L2_cp0"),        # CSA layer cp_rank=0 (128 swa + 64 compressed)
    (64, 512, 128, 319, 192, "dsv4_L2_cp1"),        # CSA layer cp_rank>0 (255 + 64)
    # --- 长序列 ---
    (64, 512, 2048, 2175, 192, "dsv4_S2048"),       # 2k seq
    # --- 同一 static 配置下的 dynamic topk ---
    (64, 512, 128, 200, 137, "dsv4_topk137"),       # topk=137, 需 pad 到 144
    (64, 512, 128, 200, 151, "dsv4_topk151"),       # topk=151, 需 pad 到 160
    (64, 512, 128, 200, 169, "dsv4_topk169"),       # topk=169, 需 pad 到 176
]

# 小 shape：覆盖 interface pad 到 16 对齐的边界，避免拖慢全量 case
_DYNAMIC_TOPK_BOUNDARY_CASES = [
    # (H, D, S, S_KV, TOPK, label)
    (64, 128, 16, 64, 1, "topk1"),
    (64, 128, 16, 64, 15, "topk15_pad16"),   # block_I-1
    (64, 128, 16, 64, 16, "topk16_exact"),   # 无需 pad
    (64, 128, 16, 64, 17, "topk17_pad32"),   # block_I+1
    (64, 128, 16, 64, 32, "topk32_exact"),
]

# 同一 H/D specialization 下换多个 padded bucket；数值 + JIT cache 复用一起验
_DYNAMIC_TOPK_REUSE_TOPKS = (1, 15, 16, 17, 33, 49, 192, 256, 312, 360)


class TestSparseAttnKernel(unittest.TestCase):
    """sparse_attn_tilelang fwd / bwd 对照 fp32 参考，覆盖多种配置。"""

    B = 1

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("cuda is required")
        torch.cuda.set_device(0)
        cls.device = torch.device(DEVICE)

    def _make_inputs(self, H, D, S, S_KV, TOPK, seed: int, drop_rate: float = 0.1):
        self.H, self.D, self.TOPK = H, D, TOPK
        g = torch.Generator(device="cpu").manual_seed(seed)
        q = torch.randn(self.B, S, self.H, self.D, generator=g)
        kv = torch.randn(self.B, S_KV, self.D, generator=g)
        sink = torch.randn(self.H, generator=g)

        idx = torch.empty(self.B, S, self.TOPK, dtype=torch.int32)
        for b in range(self.B):
            for s in range(S):
                perm = torch.randperm(S_KV,
                                      generator=g)[:self.TOPK].to(torch.int32)
                if drop_rate > 0:
                    drop = torch.rand(self.TOPK, generator=g) < drop_rate
                    drop[0] = False
                    perm[drop] = -1
                idx[b, s] = perm
        return q, kv, sink, idx

    @staticmethod
    def _ref_forward(q, kv, sink, idx, sm_scale):
        """fp32 参考：按 topk gather + 加 sink logit + softmax（丢 sink 权重）+ 加权和。

        与 modeling 的 eager_attention_forward 同源：score = q·kᵀ * scale；sink 为
        scale 空间内的裸 logit（不再乘 scale）；softmax 后丢掉 sink 列再乘 V(=KV)。
        """
        B, S, H, D = q.shape
        out = torch.zeros(B, S, H, D, dtype=torch.float32, device=q.device)
        for b in range(B):
            safe = idx[b].clamp(min=0).long()
            valid = idx[b] >= 0
            k_g = kv[b][safe]
            scores = torch.einsum(
                "shd,std->sht", q[b], k_g
            ) * sm_scale
            scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
            sink_col = sink.view(1, H, 1).expand(S, H, 1)
            logits = torch.cat([scores, sink_col], dim=-1)
            probs = torch.softmax(logits, dim=-1)[..., :-1]
            out[b] = torch.einsum("sht,std->shd", probs, k_g)
        return out

    def _assert_fwd_bwd_vs_ref(self, q, kv, sink, idx, sm_scale, label: str):
        """对照 fp32 参考校验 sparse_attn_tilelang 的 forward + backward。"""
        dev = self.device
        B, S, H, D = q.shape
        idx = idx.to(dev).contiguous()

        ref = self._ref_forward(
            q.to(dev).float(),
            kv.to(dev).float(),
            sink.to(dev).float(),
            idx,
            sm_scale,
        )
        out = sparse_attn_tilelang(
            q.to(dev, torch.bfloat16).contiguous(),
            kv.to(dev, torch.bfloat16).contiguous(),
            sink.to(dev, torch.float32).contiguous(),
            idx,
            sm_scale,
        )
        abs_d, rel_d = _worst_diff(out, ref)
        print(f"[sparse_attn:fwd {label}] max_abs={abs_d:.3e} rel={rel_d:.3e}")
        self.assertTrue(
            rel_d <= 2e-2 or abs_d <= 2e-2,
            f"fwd mismatch {label} abs={abs_d:.3e} rel={rel_d:.3e}",
        )

        g = torch.Generator(device="cpu").manual_seed(99)
        do = torch.randn(B, S, H, D, generator=g).to(dev)

        q_r = q.to(dev).float().requires_grad_(True)
        kv_r = kv.to(dev).float().requires_grad_(True)
        sink_r = sink.to(dev).float().requires_grad_(True)
        out_r = self._ref_forward(q_r, kv_r, sink_r, idx, sm_scale)
        (out_r * do).sum().backward()

        q_k = q.to(dev, torch.bfloat16).contiguous().requires_grad_(True)
        kv_k = kv.to(dev, torch.bfloat16).contiguous().requires_grad_(True)
        sink_k = sink.to(dev, torch.float32).contiguous().requires_grad_(True)
        out_k = sparse_attn_tilelang(q_k, kv_k, sink_k, idx, sm_scale)
        (out_k * do.to(torch.bfloat16)).sum().backward()

        for name, gk, gr, rtol in [
            ("dq", q_k.grad, q_r.grad, 5e-2),
            ("dkv", kv_k.grad, kv_r.grad, 5e-2),
            ("dsink", sink_k.grad, sink_r.grad, 5e-2),
        ]:
            abs_d, rel_d = _worst_diff(gk, gr)
            print(
                f"[sparse_attn:bwd:{name} {label}] max_abs={abs_d:.3e} rel={rel_d:.3e}"
            )
            self.assertTrue(
                rel_d <= rtol or abs_d <= rtol,
                f"{name} mismatch {label} abs={abs_d:.3e} rel={rel_d:.3e} (rtol={rtol})",
            )

    def test_forward(self):
        for H, D, S, S_KV, TOPK, label in _SPARSE_ATTN_CASES:
            with self.subTest(label=label):
                q, kv, sink, idx = self._make_inputs(H, D, S, S_KV, TOPK, seed=0)
                sm_scale = D**-0.5
                ref = self._ref_forward(
                    q.to(self.device).float(),
                    kv.to(self.device).float(),
                    sink.to(self.device).float(),
                    idx.to(self.device),
                    sm_scale,
                )
                out = sparse_attn_tilelang(
                    q.to(self.device, torch.bfloat16).contiguous(),
                    kv.to(self.device, torch.bfloat16).contiguous(),
                    sink.to(self.device, torch.float32).contiguous(),
                    idx.to(self.device).contiguous(),
                    sm_scale,
                )
                abs_d, rel_d = _worst_diff(out, ref)
                print(f"[sparse_attn:fwd {label}] max_abs={abs_d:.3e} rel={rel_d:.3e}")
                self.assertTrue(
                    rel_d <= 2e-2 or abs_d <= 2e-2,
                    f"fwd mismatch {label} abs={abs_d:.3e} rel={rel_d:.3e}"
                )
                del q, kv, sink, idx, ref, out
                torch.cuda.empty_cache()

    def test_backward(self):
        for H, D, S, S_KV, TOPK, label in _SPARSE_ATTN_CASES:
            with self.subTest(label=label):
                q, kv, sink, idx = self._make_inputs(H, D, S, S_KV, TOPK, seed=1)
                dev = self.device
                sm_scale = D**-0.5
                idx = idx.to(dev).contiguous()

                g = torch.Generator(device="cpu").manual_seed(99)
                do = torch.randn(self.B, S, H, D, generator=g).to(dev)

                q_r = q.to(dev).float().requires_grad_(True)
                kv_r = kv.to(dev).float().requires_grad_(True)
                sink_r = sink.to(dev).float().requires_grad_(True)
                out_r = self._ref_forward(q_r, kv_r, sink_r, idx, sm_scale)
                (out_r * do).sum().backward()

                q_k = q.to(dev, torch.bfloat16).contiguous().requires_grad_(True)
                kv_k = kv.to(dev, torch.bfloat16).contiguous().requires_grad_(True)
                sink_k = sink.to(dev, torch.float32).contiguous().requires_grad_(True)
                out_k = sparse_attn_tilelang(q_k, kv_k, sink_k, idx, sm_scale)
                (out_k * do.to(torch.bfloat16)).sum().backward()

                for name, gk, gr, rtol in [
                    ("dq", q_k.grad, q_r.grad, 5e-2),
                    ("dkv", kv_k.grad, kv_r.grad, 5e-2),
                    ("dsink", sink_k.grad, sink_r.grad, 5e-2),
                ]:
                    abs_d, rel_d = _worst_diff(gk, gr)
                    print(
                        f"[sparse_attn:bwd:{name} {label}] max_abs={abs_d:.3e} rel={rel_d:.3e}"
                    )
                    self.assertTrue(
                        rel_d <= rtol or abs_d <= rtol,
                        f"{name} mismatch {label} abs={abs_d:.3e} rel={rel_d:.3e} (rtol={rtol})",
                    )
                del q, kv, sink, idx, do, q_r, kv_r, sink_r, out_r, q_k, kv_k, sink_k, out_k
                torch.cuda.empty_cache()

    def test_dynamic_topk_boundary_fwd_bwd(self):
        for H, D, S, S_KV, TOPK, label in _DYNAMIC_TOPK_BOUNDARY_CASES:
            with self.subTest(label=label):
                q, kv, sink, idx = self._make_inputs(H, D, S, S_KV, TOPK, seed=2)
                self._assert_fwd_bwd_vs_ref(q, kv, sink, idx, D**-0.5, label)
                del q, kv, sink, idx
                torch.cuda.empty_cache()

    def test_dynamic_topk_heavy_mask_fwd_bwd(self):
        # 高比例 -1：覆盖 pad 尾部 + 有效位稀疏时的 mask 路径
        q, kv, sink, idx = self._make_inputs(
            64, 128, 16, 64, 17, seed=3, drop_rate=0.7
        )
        self._assert_fwd_bwd_vs_ref(q, kv, sink, idx, 128**-0.5, "topk17_heavy_mask")

    def test_pad_to_block_equiv(self):
        # topk=15（interface pad -1）应与显式 topk=16 且末位全 -1 bit-exact
        H, D, S, S_KV = 64, 128, 16, 64
        q, kv, sink, idx16 = self._make_inputs(
            H, D, S, S_KV, 16, seed=4, drop_rate=0.0
        )
        idx16[..., -1] = -1
        idx15 = idx16[..., :15].contiguous()
        sm_scale = D**-0.5
        dev = self.device

        out15 = sparse_attn_tilelang(
            q.to(dev, torch.bfloat16).contiguous(),
            kv.to(dev, torch.bfloat16).contiguous(),
            sink.to(dev, torch.float32).contiguous(),
            idx15.to(dev).contiguous(),
            sm_scale,
        )
        out16 = sparse_attn_tilelang(
            q.to(dev, torch.bfloat16).contiguous(),
            kv.to(dev, torch.bfloat16).contiguous(),
            sink.to(dev, torch.float32).contiguous(),
            idx16.to(dev).contiguous(),
            sm_scale,
        )
        self.assertTrue(torch.equal(out15, out16), "pad(-1) vs explicit trailing -1 mismatch")

    def test_dynamic_topk_reuses_jit_specialization(self):
        # 允许被同进程前置用例暖过同一 H/D specialization：首次至多 +1，之后不得再涨。
        before = (len(sparse_mqa_fwd._kernel_cache), len(sparse_mqa_bwd._kernel_cache))
        cached = None
        H, D, S, S_KV = 64, 128, 16, 384
        sm_scale = D**-0.5
        for topk in _DYNAMIC_TOPK_REUSE_TOPKS:
            with self.subTest(topk=topk):
                q, kv, sink, idx = self._make_inputs(H, D, S, S_KV, topk, seed=topk)
                self._assert_fwd_bwd_vs_ref(
                    q, kv, sink, idx, sm_scale, f"dyn_reuse_topk{topk}"
                )
                cache_sizes = (
                    len(sparse_mqa_fwd._kernel_cache),
                    len(sparse_mqa_bwd._kernel_cache),
                )
                if cached is None:
                    self.assertGreaterEqual(cache_sizes[0], before[0])
                    self.assertGreaterEqual(cache_sizes[1], before[1])
                    self.assertLessEqual(cache_sizes[0], before[0] + 1)
                    self.assertLessEqual(cache_sizes[1], before[1] + 1)
                    cached = cache_sizes
                else:
                    self.assertEqual(cache_sizes, cached)
                del q, kv, sink, idx
                torch.cuda.empty_cache()


class TestIndexerKernel(unittest.TestCase):
    """Lightning Indexer fwd logits 对照 fp32 参考 + top-k 一致性。"""

    # Sq / Skv：compress_ratio=4，Skv=Sq/ratio；heads%8==0；topk 为 2 的幂且 <= Skv
    SQ, B, H, DIM, SKV, RATIO, TOPK = 256, 1, 8, 128, 64, 4, 32

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("cuda is required")
        torch.cuda.set_device(0)
        cls.device = torch.device(DEVICE)

    def _make_inputs(self, seed: int):
        g = torch.Generator(device="cpu").manual_seed(seed)
        index_q = torch.randn(self.SQ, self.B, self.H, self.DIM, generator=g)
        index_k = torch.randn(self.SKV, self.B, self.DIM, generator=g)
        weights = torch.randn(self.SQ, self.B, self.H,
                              generator=g).abs()  # 正权重，贴近真实
        return index_q, index_k, weights

    def _ref_logits(self, index_q, index_k, weights):
        """fp32 参考，直接对应 modeling_deepseek_v4.py index scoring 逻辑。"""
        # 测试输入 S-first → 模型代码 B-first
        q = index_q.float().permute(1, 0, 2, 3)         # [B, Sq, H, dim]
        compressed_kv = index_k.float().transpose(0, 1)  # [B, Skv, dim]
        w = weights.float().permute(1, 0, 2)             # [B, Sq, H]

        # --- 直接对应 modeling_deepseek_v4.py L397-L416 ---
        softmax_scale = self.DIM ** -0.5
        scores = torch.matmul(q, compressed_kv.transpose(-1, -2).unsqueeze(1))  # [B, S, H, T]
        scores = F.relu(scores) * softmax_scale
        index_scores = (scores * w.unsqueeze(-1)).sum(dim=2)  # [B, S, T]

        compressed_len = compressed_kv.shape[1]
        position_ids = torch.arange(self.SQ, device=q.device).unsqueeze(0)
        causal_threshold = (position_ids + 1) // self.RATIO  # [1, Sq]
        entry_indices = torch.arange(compressed_len, device=q.device)
        future_mask = entry_indices.view(1, 1, -1) >= causal_threshold.unsqueeze(-1)  # [1, Sq, T]
        index_scores = index_scores.masked_fill(future_mask, float("-inf"))

        invalid = future_mask.squeeze(0)  # [Sq, Skv]
        return index_scores, invalid

    def test_forward_logits(self):
        index_q, index_k, weights = self._make_inputs(seed=0)
        dev = self.device
        ref, invalid = self._ref_logits(
            index_q.to(dev), index_k.to(dev), weights.to(dev)
        )

        cu_ks, cu_ke = _make_causal_cu_seqlens(
            self.SQ, self.SKV, self.RATIO, dev
        )
        logits = batched_indexer_fwd(
            index_q.to(dev, torch.bfloat16).contiguous(),
            index_k.to(dev, torch.bfloat16).contiguous(),
            weights.to(dev, torch.float32).contiguous(),
            cu_ks,
            cu_ke,
        )  # [B, Sq, Skv]

        # batched_indexer_fwd 不 clean 无效区，只比 causal 有效区
        finite = (~invalid[None]).expand_as(logits)
        abs_d, rel_d = _worst_diff(logits[finite], ref[finite])
        print(f"[indexer:fwd_logits] max_abs={abs_d:.3e} rel={rel_d:.3e}")
        self.assertTrue(
            rel_d <= 3e-2 or abs_d <= 3e-2,
            f"logits mismatch abs={abs_d:.3e} rel={rel_d:.3e}"
        )

    def test_topk_consistency(self):
        """v4_lighting_indexer 选出的 top-k 索引应与对其内部 logits 直接 topk 一致。"""
        index_q, index_k, weights = self._make_inputs(seed=2)
        dev = self.device
        q = index_q.to(dev, torch.bfloat16).contiguous()
        k = index_k.to(dev, torch.bfloat16).contiguous()
        w = weights.to(dev, torch.float32).contiguous()

        _, topk_idx = v4_lighting_indexer(q, k, w, self.RATIO, self.TOPK)

        # 复算内部 logits（确定性，同输入 → 同 kernel 输出），自己做 topk + clean 无效区
        cu_ks, cu_ke = _make_causal_cu_seqlens(
            self.SQ, self.SKV, self.RATIO, dev
        )
        logits = batched_indexer_fwd(q, k, w, cu_ks, cu_ke)
        s_pos = torch.arange(self.SQ, device=dev)
        ke = ((s_pos + 1) // self.RATIO).clamp(max=self.SKV)
        kv_pos = torch.arange(self.SKV, device=dev)
        invalid = kv_pos[None, :] >= ke[:, None]
        logits = logits.masked_fill(invalid[None], float("-inf"))

        actual_topk = min(self.TOPK, self.SKV)
        sc, idx_ref = torch.topk(logits, actual_topk, dim=-1)
        idx_ref = idx_ref.to(torch.int32).masked_fill(sc == float("-inf"), -1)

        self.assertEqual(tuple(topk_idx.shape), tuple(idx_ref.shape))
        mism = (topk_idx.to(torch.int32) != idx_ref).float().mean().item()
        print(f"[indexer:topk] index-mismatch fraction={mism:.3e}")
        self.assertEqual(
            mism, 0.0, f"top-k indices mismatch fraction={mism:.3e}"
        )


if __name__ == "__main__":
    unittest.main()
