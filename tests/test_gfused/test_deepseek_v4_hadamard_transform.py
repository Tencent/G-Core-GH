# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""DeepSeek-V4 Lightning Indexer Hadamard rotation + FP4 QAT (``fp4_qat_indexer``).

Two groups of tests:

(a) ``test_rotation_*`` — pure-tensor math: the rotation is orthogonal and
    cancels out of the indexer's Q·K score.
(b) ``test_fp4_error_*`` — the rotation shrinks FP4 block-quantization error as
    channel outliers grow.

Usage::

    cd /work/wepsdl/gcore-dev
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/gcore-dev/tests/test_gpatch_v4:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    pytest -v -s --timeout=3600 tests/test_gfused/test_deepseek_v4_hadamard_transform.py
"""

import unittest

import pytest
import torch
import torch.nn.functional as F

try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    pytest.skip("fast_hadamard_transform is not installed", allow_module_level=True)

from gpatch_v4.models.deepseek_v4.kernel.hadamard_transform import rotate_activation

INDEX_HEAD_DIM = 128
FP4_BLOCK = 32
MIN_TOPK_OVERLAP = 0.99


# ---------------------------------------------------------------------------
# (a) 数学等价：纯张量
# ---------------------------------------------------------------------------


def _sylvester_hadamard(dim: int, device, dtype=torch.float64) -> torch.Tensor:
    """Normalized Sylvester Hadamard matrix, independent of the CUDA kernel."""
    h = torch.ones(1, 1, device=device, dtype=dtype)
    while h.shape[0] < dim:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    assert h.shape[0] == dim, f"dim {dim} is not a power of 2"
    return h * dim**-0.5


def _index_scores(q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor,
                  softmax_scale: float,
                  compute_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``DeepseekV4Indexer`` eager-branch score, verbatim in formula.

    Parameters
    ----------
    q : torch.Tensor
        Shape ``[B, S, H, D]``.
    k : torch.Tensor
        Shape ``[B, T, D]``.
    weights : torch.Tensor
        Shape ``[B, S, H]``.
    compute_dtype : torch.dtype
        Accumulation dtype. Defaults to fp32 to mirror the model's ``.float()``;
        pass fp64 to separate the rotation's algebraic error from fp32 rounding.

    Returns
    -------
    torch.Tensor
        Shape ``[B, S, T]``.
    """
    scores = torch.matmul(
        q.to(compute_dtype), k.transpose(-1, -2).to(compute_dtype).unsqueeze(1),
    )
    scores = F.relu(scores) * softmax_scale
    return (scores * weights.unsqueeze(-1).to(compute_dtype)).sum(dim=2)


def _topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    """Mean per-row fraction of `a`'s top-`k` picks that also appear in `b`'s."""
    ia = a.topk(k, dim=-1).indices
    ib = b.topk(k, dim=-1).indices
    n_entries = a.shape[-1]
    occ = torch.zeros(*ib.shape[:-1], n_entries, dtype=torch.bool, device=b.device)
    occ.scatter_(-1, ib, True)
    return occ.gather(-1, ia).float().mean().item()


def test_rotation_matrix_is_orthogonal_and_involutory():
    """``H/sqrt(d)`` 正交且对合——反向传播复用同一变换的前提。"""
    h = _sylvester_hadamard(INDEX_HEAD_DIM, device="cpu")
    eye = torch.eye(INDEX_HEAD_DIM, dtype=torch.float64)
    assert (h @ h.T - eye).abs().max().item() < 1e-12
    assert (h @ h - eye).abs().max().item() < 1e-12


def test_rotate_activation_matches_sylvester_reference():
    """kernel 与 eager 实现一致。"""
    torch.manual_seed(0)
    x = torch.randn(64, INDEX_HEAD_DIM, device="cuda", dtype=torch.bfloat16)

    got = rotate_activation(x).float()
    h = _sylvester_hadamard(INDEX_HEAD_DIM, device="cuda", dtype=torch.float32)
    expected = x.float() @ h

    rel = (got - expected).norm() / expected.norm()
    print(f"\n  rotate_activation vs Sylvester ref: rel_l2={rel.item():.3e}")
    assert rel.item() < 2e-2, f"kernel disagrees with the reference matrix: {rel.item():.3e}"


def test_rotation_cancels_in_index_scores_fp64():
    """fp64 下 HT 在 Q·K 里精确抵消，误差只剩浮点舍入。"""
    torch.manual_seed(0)
    b, s, hh, t, d = 1, 1024, 64, 256, INDEX_HEAD_DIM
    q = torch.randn(b, s, hh, d, dtype=torch.float64)
    k = torch.randn(b, t, d, dtype=torch.float64)
    w = torch.randn(b, s, hh, dtype=torch.float64).abs()
    h = _sylvester_hadamard(d, device="cpu")
    scale = d**-0.5

    ref = _index_scores(q, k, w, scale, compute_dtype=torch.float64)
    rot = _index_scores(q @ h, k @ h, w, scale, compute_dtype=torch.float64)

    rel = ((rot - ref).norm() / ref.norm()).item()
    print(f"\n  fp64 index_scores rel_l2={rel:.3e}")
    assert rel < 1e-10, f"HT should cancel exactly in fp64, got {rel:.3e}"


def test_rotation_preserves_index_scores_bf16():
    """走真实 kernel + bf16：分数近似不变、top-k 高度重合。

    bf16 下不是逐比特等价（旋转后要重新舍入才进 GEMM），所以只能要求相对误差
    和 top-k 重合率，不能要求精确相等。
    """
    torch.manual_seed(0)
    b, s, hh, t, d = 1, 1024, 64, 256, INDEX_HEAD_DIM
    q = torch.randn(b, s, hh, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, t, d, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(b, s, hh, device="cuda", dtype=torch.float32).abs()
    scale = d**-0.5

    ref = _index_scores(q, k, w, scale, compute_dtype=torch.bfloat16)
    rot = _index_scores(rotate_activation(q), rotate_activation(k), w, scale, compute_dtype=torch.bfloat16)

    rel = ((rot - ref).norm() / ref.norm()).item()
    overlap = _topk_overlap(rot, ref, k=32)
    print(f"\n  bf16 index_scores rel_l2={rel:.3e}  top32_overlap={overlap:.4f}")
    assert rel < 1e-2, f"bf16 rel_l2 {rel:.3e} too large for a math-equivalent rotation"
    assert overlap > MIN_TOPK_OVERLAP, f"top-k overlap {overlap:.4f} too low"


def test_rotate_activation_rejects_non_power_of_two_dim():
    """非 2 的幂必须报错。

    ``fast_hadamard_transform`` 对非 2 的幂会静默 zero-pad 到下一个 2 的幂再把
    输出截回去，得到的映射既不正交也不保范，内积会失真好几个数量级——静默错，
    必须在入口拦住。
    """
    x = torch.randn(8, 192, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="power of 2"):
        rotate_activation(x)


def test_rotate_activation_rejects_non_bf16():
    x = torch.randn(8, INDEX_HEAD_DIM, device="cuda", dtype=torch.float32)
    with pytest.raises(AssertionError, match="bfloat16"):
        rotate_activation(x)


# ---------------------------------------------------------------------------
# (b) FP4 量化误差随离群程度的下降
# ---------------------------------------------------------------------------


def _make_outlier_tensor(rows: int, dim: int, mag: float, n_outlier: int,
                         seed: int) -> torch.Tensor:
    """标准正态 + `n_outlier` 个通道整体放大 `mag` 倍。"""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, dim, generator=gen)
    channels = torch.randperm(dim, generator=gen)[:n_outlier]
    x[:, channels] *= mag
    return x.to(device="cuda", dtype=torch.bfloat16)


def _block_amax_spread(x: torch.Tensor, block: int = FP4_BLOCK) -> float:
    """每行内各量化块 amax 的不均匀度 ``max/mean``，1.0 = 完全均匀。

    这是旋转要解决的问题本身：一个离群通道会独占它所在块的 scale，把同块其他
    元素压到 FP4 网格底部。
    """
    block_amax = x.float().unflatten(-1, (-1, block)).abs().amax(-1)
    return (block_amax.amax(-1) / block_amax.mean(-1)).mean().item()


@pytest.mark.parametrize("mag", [1.0, 3.0, 10.0, 30.0])
def test_fp4_error_rotation_flattens_block_amax(mag: float):
    """旋转把各量化块的 amax 拉平；离群越强收益越大。"""
    x = _make_outlier_tensor(512, INDEX_HEAD_DIM, mag, n_outlier=4, seed=0)

    spread_raw = _block_amax_spread(x)
    spread_rot = _block_amax_spread(rotate_activation(x))
    print(f"\n  mag={mag:5.1f}  block-amax spread: raw={spread_raw:.3f} -> rot={spread_rot:.3f}")

    assert spread_rot < 1.5, f"rotated spread {spread_rot:.3f} not flat"
    if mag >= 10.0:
        assert spread_raw > 2.0, (
            f"mag={mag} should produce an uneven raw spread, got {spread_raw:.3f}; "
            f"the outlier fixture is not doing its job"
        )
        assert spread_rot < spread_raw * 0.75, (
            f"rotation barely helped: {spread_raw:.3f} -> {spread_rot:.3f}"
        )


@pytest.mark.parametrize("mag", [1.0, 3.0, 10.0, 30.0])
def test_fp4_error_rotation_improves_score_fidelity(mag: float):
    """离群档位下，先旋转再 FP4 的 Q·K 分数比直接 FP4 更接近 fp32 参考。

    旋转在分数里精确抵消，所以两条路径的差别只来自量化误差；离群越强，未旋转
    的 per-block scale 被独占得越厉害，分数越差。
    """
    from gpatch_v4.kernel.quantize.qat import fp4_simulate_qat

    d, n_heads, n_keys = INDEX_HEAD_DIM, 8, 256
    q = _make_outlier_tensor(64 * n_heads, d, mag, n_outlier=4, seed=1)
    k = _make_outlier_tensor(n_keys, d, mag, n_outlier=4, seed=2)
    q = q.view(1, 64, n_heads, d)
    k = k.view(1, n_keys, d)
    w = torch.ones(1, 64, n_heads, device="cuda", dtype=torch.float32)
    scale = d**-0.5

    ref = _index_scores(q, k, w, scale)
    plain = _index_scores(
        fp4_simulate_qat(q.contiguous(), FP4_BLOCK),
        fp4_simulate_qat(k.contiguous(), FP4_BLOCK),
        w, scale,
    )
    rotated = _index_scores(
        fp4_simulate_qat(rotate_activation(q), FP4_BLOCK),
        fp4_simulate_qat(rotate_activation(k), FP4_BLOCK),
        w, scale,
    )

    err_plain = ((plain - ref).norm() / ref.norm()).item()
    err_rot = ((rotated - ref).norm() / ref.norm()).item()
    ov_plain = _topk_overlap(plain, ref, k=32)
    ov_rot = _topk_overlap(rotated, ref, k=32)
    print(
        f"\n  mag={mag:5.1f}  rel_l2: plain={err_plain:.4f} rot={err_rot:.4f}"
        f"   top32_overlap: plain={ov_plain:.4f} rot={ov_rot:.4f}"
    )

    if mag >= 10.0:
        assert err_rot < err_plain, (
            f"rotation should cut FP4 score error at mag={mag}: "
            f"plain={err_plain:.4f} rot={err_rot:.4f}"
        )
        assert ov_rot > ov_plain, (
            f"rotation should improve top-k fidelity at mag={mag}: "
            f"plain={ov_plain:.4f} rot={ov_rot:.4f}"
        )
    else:
        # 无离群时旋转是中性的，只要求不退化。
        assert ov_rot > ov_plain - 0.05, (
            f"rotation regressed on a benign distribution: "
            f"plain={ov_plain:.4f} rot={ov_rot:.4f}"
        )


if __name__ == "__main__":
    unittest.main()
