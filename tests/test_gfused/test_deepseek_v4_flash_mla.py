# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""FlashMLA sparse-attn wrapper：fwd + cuDNN DSA bwd vs fp32 参考。"""

import unittest

import pytest
import torch

pytest.importorskip("flash_mla")

from gpatch_v4.models.deepseek_v4.kernel.flash_mla_wrapper import (
    sparse_attn_flash_mla,
)

DEVICE = "cuda:0"

# (B, H, D, S, S_KV, TOPK, label)
# 取自 test_deepseek_v4_kernel_ops 的 DSV4-Flash 真实参数，外加 wrapper 特有的 B>1 / 不对齐 topk。
# 不跑 miles H=512 / S=4096：FlashMLA d_qk 不是 256，且长序列会拖慢 gfused 套件。
_CASES = [
    (1, 64, 512, 128, 128, 128, "dsv4_L0_cp0"),
    (1, 64, 512, 128, 255, 128, "dsv4_L0_cp1"),
    (1, 64, 512, 128, 192, 192, "dsv4_L2_cp0"),
    (1, 64, 512, 64, 80, 80, "topk80_pad"),
    (2, 64, 512, 64, 64, 64, "B2_bmajor"),
]


def _worst_diff(actual: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """返回 (max_abs_diff, max_rel_diff)，rel 以 ref 的最大绝对值为分母。"""
    a = actual.detach().float()
    r = ref.detach().float()
    abs_d = (a - r).abs().max().item()
    rel_d = abs_d / max(r.abs().max().item(), 1e-12)
    return abs_d, rel_d


class TestFlashMLASparseAttn(unittest.TestCase):
    """sparse_attn_flash_mla fwd / bwd 对照 fp32 参考。"""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("cuda is required")
        try:
            from cudnn import DSA  # noqa: F401
        except ImportError as e:
            raise unittest.SkipTest(f"cudnn.DSA unavailable: {e}")
        torch.cuda.set_device(0)
        cls.device = torch.device(DEVICE)

    def _make_inputs(self, B, H, D, S, S_KV, TOPK, seed: int, drop_rate: float = 0.1):
        g = torch.Generator(device="cpu").manual_seed(seed)
        q = torch.randn(B, S, H, D, generator=g)
        kv = torch.randn(B, S_KV, D, generator=g)
        sink = torch.randn(H, generator=g)

        idx = torch.empty(B, S, TOPK, dtype=torch.int32)
        for b in range(B):
            for s in range(S):
                perm = torch.randperm(S_KV, generator=g)[:TOPK].to(torch.int32)
                if drop_rate > 0:
                    drop = torch.rand(TOPK, generator=g) < drop_rate
                    drop[0] = False
                    perm[drop] = -1
                idx[b, s] = perm
        return q, kv, sink, idx

    @staticmethod
    def _ref_forward(q, kv, sink, idx, sm_scale):
        """fp32 参考：按 topk gather + 加 sink logit + softmax（丢 sink 权重）+ 加权和。

        与 ``test_deepseek_v4_kernel_ops.TestSparseAttnKernel._ref_forward`` 同源。
        """
        B, S, H, D = q.shape
        out = torch.zeros(B, S, H, D, dtype=torch.float32, device=q.device)
        for b in range(B):
            safe = idx[b].clamp(min=0).long()
            valid = idx[b] >= 0
            k_g = kv[b][safe]
            scores = torch.einsum("shd,std->sht", q[b], k_g) * sm_scale
            scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
            sink_col = sink.view(1, H, 1).expand(S, H, 1)
            logits = torch.cat([scores, sink_col], dim=-1)
            probs = torch.softmax(logits, dim=-1)[..., :-1]
            out[b] = torch.einsum("sht,std->shd", probs, k_g)
        return out

    def _assert_fwd_bwd_vs_ref(self, q, kv, sink, idx, sm_scale, label: str):
        """对照 fp32 参考校验 FlashMLA wrapper 的 forward + backward。"""
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
        out = sparse_attn_flash_mla(
            q.to(dev, torch.bfloat16).contiguous(),
            kv.to(dev, torch.bfloat16).contiguous(),
            sink.to(dev, torch.float32).contiguous(),
            idx,
            sm_scale,
        )
        abs_d, rel_d = _worst_diff(out, ref)
        print(f"[flash_mla:fwd {label}] max_abs={abs_d:.3e} rel={rel_d:.3e}")
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
        out_k = sparse_attn_flash_mla(q_k, kv_k, sink_k, idx, sm_scale)
        (out_k * do.to(torch.bfloat16)).sum().backward()

        for name, gk, gr, rtol in [
            ("dq", q_k.grad, q_r.grad, 5e-2),
            ("dkv", kv_k.grad, kv_r.grad, 5e-2),
            ("dsink", sink_k.grad, sink_r.grad, 5e-2),
        ]:
            abs_d, rel_d = _worst_diff(gk, gr)
            print(
                f"[flash_mla:bwd:{name} {label}] max_abs={abs_d:.3e} rel={rel_d:.3e}"
            )
            self.assertTrue(
                rel_d <= rtol or abs_d <= rtol,
                f"{name} mismatch {label} abs={abs_d:.3e} rel={rel_d:.3e} (rtol={rtol})",
            )

    def test_forward_backward(self):
        for B, H, D, S, S_KV, TOPK, label in _CASES:
            with self.subTest(label=label):
                q, kv, sink, idx = self._make_inputs(
                    B, H, D, S, S_KV, TOPK, seed=0
                )
                self._assert_fwd_bwd_vs_ref(q, kv, sink, idx, D**-0.5, label)
                del q, kv, sink, idx
                torch.cuda.empty_cache()

    def test_heavy_mask_fwd_bwd(self):
        q, kv, sink, idx = self._make_inputs(
            1, 64, 512, 64, 80, 80, seed=3, drop_rate=0.7
        )
        self._assert_fwd_bwd_vs_ref(q, kv, sink, idx, 512**-0.5, "topk80_heavy_mask")

    def test_b2_matches_per_batch(self):
        """B-major flatten 不得串 batch：B=2 应与逐条 B=1 拼接 bit-exact。"""
        B, H, D, S, S_KV, TOPK = 2, 64, 512, 64, 64, 64
        q, kv, sink, idx = self._make_inputs(B, H, D, S, S_KV, TOPK, seed=1)
        dev = self.device
        sm_scale = D**-0.5
        q = q.to(dev, torch.bfloat16).contiguous()
        kv = kv.to(dev, torch.bfloat16).contiguous()
        sink = sink.to(dev, torch.float32).contiguous()
        idx = idx.to(dev).contiguous()

        out_b2 = sparse_attn_flash_mla(q, kv, sink, idx, sm_scale)
        outs = [
            sparse_attn_flash_mla(
                q[b : b + 1], kv[b : b + 1], sink, idx[b : b + 1], sm_scale
            )
            for b in range(B)
        ]
        out_cat = torch.cat(outs, dim=0)
        abs_d, _ = _worst_diff(out_b2, out_cat)
        print(f"[flash_mla:B2_vs_B1] max_abs={abs_d:.3e}")
        self.assertEqual(abs_d, 0.0, f"B=2 vs per-batch B=1 mismatch abs={abs_d:.3e}")

    def test_unaligned_topk_pad_equiv(self):
        """topk=80（wrapper pad -1）应与显式 topk=128 且尾部全 -1 bit-exact。"""
        H, D, S, S_KV = 64, 512, 64, 80
        q, kv, sink, idx80 = self._make_inputs(
            1, H, D, S, S_KV, 80, seed=4, drop_rate=0.0
        )
        idx128 = torch.nn.functional.pad(idx80, (0, 48), value=-1)
        sm_scale = D**-0.5
        dev = self.device

        q_bf = q.to(dev, torch.bfloat16).contiguous()
        kv_bf = kv.to(dev, torch.bfloat16).contiguous()
        sink_f = sink.to(dev, torch.float32).contiguous()
        out80 = sparse_attn_flash_mla(
            q_bf, kv_bf, sink_f, idx80.to(dev).contiguous(), sm_scale
        )
        out128 = sparse_attn_flash_mla(
            q_bf, kv_bf, sink_f, idx128.to(dev).contiguous(), sm_scale
        )
        self.assertTrue(
            torch.equal(out80, out128),
            "unaligned topk pad(-1) vs explicit trailing -1 mismatch",
        )


if __name__ == "__main__":
    unittest.main()
