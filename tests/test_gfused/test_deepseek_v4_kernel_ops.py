# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
# nrwu@tencent.com
"""DeepSeek-V4 TileLang kernel op 单元测试（从 miles 移植后的 cleanup 验证）。

覆盖两个 op：
  * sparse_attn_tilelang  —— 稀疏 MQA attention（fwd + bwd），对照 fp32 PyTorch 参考实现
  * v4_lighting_indexer / batched_indexer_fwd —— Lightning Indexer 打分 + top-k

判据：fused kernel（bf16 计算 + fp32 累加）与纯 fp32 参考在 bf16 容差内一致。
单进程单卡，需要 GPU（tilelang JIT 编译真实 CUDA kernel）。

Usage::

    cd /work/wepsdl/gcore-dev
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$PYTHONPATH"
    pytest -v -s --timeout=1800 tests/test_gfused/test_deepseek_v4_kernel_ops.py
"""

import unittest

import torch

from gpatch_v4.models.deepseek_v4.kernel.tilelang_indexer import v4_lighting_indexer
from gpatch_v4.models.deepseek_v4.kernel.tilelang_indexer_fwd import (
    _make_causal_cu_seqlens,
    batched_indexer_fwd,
)
from gpatch_v4.models.deepseek_v4.kernel.tilelang_sparse_mla import sparse_attn_tilelang

DEVICE = "cuda:0"


def _worst_diff(actual: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """返回 (max_abs_diff, max_rel_diff)，rel 以 ref 的最大绝对值为分母。"""
    a = actual.detach().float()
    r = ref.detach().float()
    abs_d = (a - r).abs().max().item()
    rel_d = abs_d / max(r.abs().max().item(), 1e-12)
    return abs_d, rel_d


class TestSparseAttnKernel(unittest.TestCase):
    """sparse_attn_tilelang fwd / bwd 对照 fp32 参考。"""

    # 形状：单 KV head MQA，head_dim=512（V4），topk 对齐 block_I=64
    B, S, S_KV, H, D, TOPK = 1, 64, 128, 16, 512, 64

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("cuda is required")
        torch.cuda.set_device(0)
        cls.device = torch.device(DEVICE)
        cls.sm_scale = cls.D**-0.5

    def _make_inputs(self, seed: int):
        """造 q / kv / sink / topk_idxs；topk_idxs 每行取不重复索引并随机插 -1。"""
        g = torch.Generator(device="cpu").manual_seed(seed)
        q = torch.randn(self.B, self.S, self.H, self.D, generator=g)
        kv = torch.randn(self.B, self.S_KV, self.D, generator=g)
        sink = torch.randn(self.H, generator=g)

        idx = torch.empty(self.B, self.S, self.TOPK, dtype=torch.int32)
        for b in range(self.B):
            for s in range(self.S):
                perm = torch.randperm(self.S_KV,
                                      generator=g)[:self.TOPK].to(torch.int32)
                # 随机把约 10% 置 -1（无效 slot），但至少保留一个有效
                drop = torch.rand(self.TOPK, generator=g) < 0.1
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
        topk = idx.shape[-1]
        out = torch.zeros(B, S, H, D, dtype=torch.float32, device=q.device)
        for b in range(B):
            safe = idx[b].clamp(min=0).long()  # [S, topk]
            valid = idx[b] >= 0  # [S, topk]
            k_g = kv[b][safe]  # [S, topk, D]
            scores = torch.einsum(
                "shd,std->sht", q[b], k_g
            ) * sm_scale  # [S, H, topk]
            scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
            sink_col = sink.view(1, H, 1).expand(S, H, 1)
            logits = torch.cat([scores, sink_col], dim=-1)  # [S, H, topk+1]
            probs = torch.softmax(logits, dim=-1)[..., :-1]  # 丢 sink 列
            out[b] = torch.einsum("sht,std->shd", probs, k_g)
        return out

    def test_forward(self):
        q, kv, sink, idx = self._make_inputs(seed=0)
        dev = self.device

        ref = self._ref_forward(
            q.to(dev).float(),
            kv.to(dev).float(),
            sink.to(dev).float(),
            idx.to(dev),
            self.sm_scale,
        )
        out = sparse_attn_tilelang(
            q.to(dev, torch.bfloat16).contiguous(),
            kv.to(dev, torch.bfloat16).contiguous(),
            sink.to(dev, torch.float32).contiguous(),
            idx.to(dev).contiguous(),
            self.sm_scale,
        )
        abs_d, rel_d = _worst_diff(out, ref)
        print(f"[sparse_attn:fwd] max_abs={abs_d:.3e} rel={rel_d:.3e}")
        self.assertTrue(
            rel_d <= 2e-2 or abs_d <= 2e-2,
            f"fwd mismatch abs={abs_d:.3e} rel={rel_d:.3e}"
        )

    def test_backward(self):
        q, kv, sink, idx = self._make_inputs(seed=1)
        dev = self.device
        idx = idx.to(dev).contiguous()

        g = torch.Generator(device="cpu").manual_seed(99)
        do = torch.randn(self.B, self.S, self.H, self.D, generator=g).to(dev)

        # 参考路径（fp32 leaf）
        q_r = q.to(dev).float().requires_grad_(True)
        kv_r = kv.to(dev).float().requires_grad_(True)
        sink_r = sink.to(dev).float().requires_grad_(True)
        out_r = self._ref_forward(q_r, kv_r, sink_r, idx, self.sm_scale)
        (out_r * do).sum().backward()

        # kernel 路径（bf16 leaf，与参考同值）
        q_k = q.to(dev, torch.bfloat16).contiguous().requires_grad_(True)
        kv_k = kv.to(dev, torch.bfloat16).contiguous().requires_grad_(True)
        sink_k = sink.to(dev, torch.float32).contiguous().requires_grad_(True)
        out_k = sparse_attn_tilelang(q_k, kv_k, sink_k, idx, self.sm_scale)
        (out_k * do.to(torch.bfloat16)).sum().backward()

        for name, gk, gr, rtol in [
            ("dq", q_k.grad, q_r.grad, 5e-2),
            ("dkv", kv_k.grad, kv_r.grad, 5e-2),
            ("dsink", sink_k.grad, sink_r.grad, 5e-2),
        ]:
            abs_d, rel_d = _worst_diff(gk, gr)
            print(
                f"[sparse_attn:bwd:{name}] max_abs={abs_d:.3e} rel={rel_d:.3e}"
            )
            self.assertTrue(
                rel_d <= rtol or abs_d <= rtol,
                f"{name} mismatch abs={abs_d:.3e} rel={rel_d:.3e} (rtol={rtol})",
            )


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
        """fp32 参考：logits[b,s,kv] = Σ_h relu(q·k) * w；causal: kv < (s+1)//ratio。"""
        qf = index_q.float()  # [Sq, B, H, dim]
        kf = index_k.float()  # [Skv, B, dim]
        dot = torch.einsum("sbhd,kbd->sbhk", qf, kf)  # [Sq, B, H, Skv]
        relu = dot.clamp(min=0)
        logits = torch.einsum(
            "sbhk,sbh->bsk", relu, weights.float()
        )  # [B, Sq, Skv]
        # causal mask（与 _make_causal_cu_seqlens 一致：ke=(s+1)//ratio，capped Skv）
        s_pos = torch.arange(self.SQ, device=logits.device)
        ke = ((s_pos + 1) // self.RATIO).clamp(max=self.SKV)  # [Sq]
        kv_pos = torch.arange(self.SKV, device=logits.device)
        invalid = kv_pos[None, :] >= ke[:, None]  # [Sq, Skv]
        logits = logits.masked_fill(invalid[None], float("-inf"))
        return logits, invalid

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
