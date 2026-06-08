"""
Fused HCA attention forward: correctness + benchmark vs sparse_mqa_fwd.

我们的 kernel 把 sliding-window attn + compressed attn + attn_sink 三段
fuse 成一个 two-phase online-softmax kernel，输入 KV 和 compressed KV 分开传，
不需要 concat 也不需要 top-K 计算。compressed causal threshold 在 kernel 内
直接算 ``(q_pos + 1) // compress_rate``，不需要额外传入。

Usage::

    cd {gw_dir}
    pytest -s tests/test_gfused/test_myfa_hca_fwd.py -v
    pytest -s tests/test_gfused/test_myfa_hca_fwd.py -k bench -v
"""

import math
import sys

import pytest
import torch

sys.path.insert(0, "gpatch_v4/models/deepseek_v4/kernel")
from myfa_hca import myfa_hca, myfa_hca_fwd
from tilelang_sparse_mla_fwd import sparse_mqa_fwd_interface


# ---------------------------------------------------------------------------
# eager reference
# ---------------------------------------------------------------------------


def ref_hca_attn(q, kv, ckv, attn_sink, scaling, sliding_window, compress_rate):
    """Eager reference: sliding-window + compressed + sink, full materialized.

    q: [B, H, LQ, D]   kv: [B, H, LKV, D] (K=V)   ckv: [B, 1, LCKV, D] (K=V)
    attn_sink: [H] fp32
    """
    B, H, LQ, D = q.shape
    LKV = kv.shape[2]
    LCKV = ckv.shape[2]

    # 1. sliding window scores
    swa_scores = torch.matmul(q.float(), kv.float().transpose(2, 3)) * scaling
    q_pos = torch.arange(LQ, device=q.device).view(1, 1, -1, 1)
    k_pos = torch.arange(LKV, device=q.device).view(1, 1, 1, -1)
    swa_mask = (k_pos > q_pos) | (q_pos - k_pos >= sliding_window)
    swa_scores = swa_scores.masked_fill(swa_mask, float("-inf"))

    # 2. compressed scores — threshold = (q_pos + 1) // compress_rate
    ckv_scores = torch.matmul(q.float(), ckv.float().transpose(2, 3)) * scaling
    entry_idx = torch.arange(LCKV, device=q.device).view(1, 1, 1, -1)
    ct = (q_pos.squeeze(3) + 1) // compress_rate  # [1, 1, LQ]
    ckv_mask = entry_idx >= ct.unsqueeze(-1)
    ckv_scores = ckv_scores.masked_fill(ckv_mask, float("-inf"))

    # 3. concat + sink
    sink_col = attn_sink.view(1, H, 1, 1).expand(B, H, LQ, 1).float()
    all_scores = torch.cat([swa_scores, ckv_scores, sink_col], dim=-1)
    p = torch.softmax(all_scores, dim=-1)

    # 4. output = P_swa @ kv + P_ckv @ ckv  (sink 列 value=0)
    o = (torch.matmul(p[:, :, :, :LKV], kv.float())
         + torch.matmul(p[:, :, :, LKV:LKV + LCKV], ckv.float()))
    return o.to(q.dtype)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_inputs(B, H, LQ, D, compress_rate):
    rng = torch.Generator(device="cuda").manual_seed(42)
    q = torch.randn(B, H, LQ, D, generator=rng, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(B, H, LQ, D, generator=rng, device="cuda", dtype=torch.bfloat16)
    LCKV = LQ // compress_rate
    ckv = torch.randn(B, 1, LCKV, D, generator=rng, device="cuda", dtype=torch.bfloat16)
    attn_sink = torch.randn(H, device="cuda", dtype=torch.float32) * 0.1
    scaling = 1.0 / math.sqrt(D)
    return q, kv, ckv, attn_sink, scaling


def _check(tag, a, b, atol_avg=0.01, atol_max=0.05):
    diff = (a.float() - b.float()).abs()
    avg = diff.mean().item()
    mx = diff.max().item()
    print(f"  {tag} | avg {avg:.6e} | max {mx:.6e}")
    assert avg < atol_avg, f"{tag} avg {avg} >= {atol_avg}"
    assert mx < atol_max, f"{tag} max {mx} >= {atol_max}"


def _bench(fn, n_warmup=10, n_iter=20):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(n_iter):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / n_iter


# ---------------------------------------------------------------------------
# correctness
# ---------------------------------------------------------------------------

# (B, H, LQ, D, compress_rate, sliding_window)
CORRECTNESS_CASES = [
    (1, 4, 256, 128, 64, 128),
    (2, 4, 512, 128, 128, 128),
    (1, 8, 1024, 256, 128, 128),
    (2, 4, 4096, 256, 128, 128),
]


@pytest.mark.parametrize("B,H,LQ,D,m,W", CORRECTNESS_CASES)
@torch.no_grad()
def test_correctness(B, H, LQ, D, m, W):
    q, kv, ckv, attn_sink, scaling = _make_inputs(B, H, LQ, D, m)
    ref_o = ref_hca_attn(q, kv, ckv, attn_sink, scaling, W, m)
    my_o = myfa_hca(q, kv, ckv, attn_sink, W, m)
    tag = f"B={B} H={H} LQ={LQ} D={D} m={m} W={W}"
    _check(tag, my_o, ref_o, atol_avg=0.02, atol_max=0.1)


# ---------------------------------------------------------------------------
# helpers for sparse_mqa_fwd comparison
# ---------------------------------------------------------------------------


def _make_sparse_topk_idxs(LQ, LCKV, sliding_window, compress_rate):
    """Build topk indices: sliding-window + compressed entries.

    sparse_mqa_fwd 把所有 token 看成 [KV_concat] = [kv | ckv]。
    topk_idxs[1, t, :] = window indices ∪ compressed indices, -1 填充。
    batch 维度不影响 index 结构（position = index），返回 [1, LQ, topk]。
    """
    topk = sliding_window + LCKV
    idxs = torch.full((1, LQ, topk), -1, device="cuda", dtype=torch.int32)

    for t in range(LQ):
        w_start = max(0, t - sliding_window + 1)
        w_end = t + 1
        w_len = w_end - w_start
        idxs[0, t, :w_len] = torch.arange(w_start, w_end, device="cuda", dtype=torch.int32)
        c = (t + 1) // compress_rate
        if c > 0:
            idxs[0, t, w_len:w_len + c] = torch.arange(
                LQ, LQ + c, device="cuda", dtype=torch.int32,
            )
    return idxs


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------

# (B, H, LQ, D, compress_rate, sliding_window)
BENCH_CASES = [
    (2, 128, 4096, 256, 128, 128),
    (2, 128, 8192, 256, 128, 128),
]


@pytest.mark.parametrize("B,H,LQ,D,m,W", BENCH_CASES)
@torch.no_grad()
def test_bench(B, H, LQ, D, m, W):
    q, kv, ckv, attn_sink, scaling = _make_inputs(B, H, LQ, D, m)
    LCKV = LQ // m

    # 1. 精度检查
    ref_o = ref_hca_attn(q, kv, ckv, attn_sink, scaling, W, m)
    my_o = myfa_hca(q, kv, ckv, attn_sink, W, m)
    tag = f"B={B} H={H} LQ={LQ} D={D} m={m} W={W}"
    _check(tag, my_o, ref_o, atol_avg=0.02, atol_max=0.15)

    # ---- 准备 sparse 路径的数据 ----
    # sparse_mqa_fwd 需要 MQA: kv_concat [B, LQ+LCKV, D]
    # DSV4 的 sliding-window KV 是 multi-head，不能直接喂给 sparse_mqa_fwd
    # 这里用 head=0 slice 做近似对比（只比 kernel 速度，不比精度）
    kv_concat = torch.cat([kv[:, 0, :, :], ckv.squeeze(1)], dim=1).contiguous()
    q_bshd = q.transpose(1, 2).contiguous()
    topk_idxs = _make_sparse_topk_idxs(LQ, LCKV, W, m).expand(B, -1, -1).contiguous()

    # ---- 编译 kernels ----
    my_kernel = myfa_hca_fwd.compile(
        H=H, D=D, scaling=scaling, sliding_window=W, compress_rate=m,
    )

    # ---- benchmark ----
    def run_ours():
        my_kernel(q, kv, ckv, attn_sink)

    def run_topk():
        _make_sparse_topk_idxs(LQ, LCKV, W, m)

    def run_sparse():
        sparse_mqa_fwd_interface(q_bshd, kv_concat, attn_sink, topk_idxs, sm_scale=scaling)

    my_ms = _bench(run_ours)
    topk_ms = _bench(run_topk)
    sp_ms = _bench(run_sparse)

    print(f"\n{'='*60}")
    print(f"  {tag}")
    print(f"  ours (fused 2-phase)   : {my_ms:8.4f} ms")
    print(f"  sparse kernel only     : {sp_ms:8.4f} ms")
    print(f"  topk idx computation   : {topk_ms:8.4f} ms")
    print(f"  sparse total (k+topk)  : {sp_ms + topk_ms:8.4f} ms")
    if my_ms > 0:
        print(f"  speedup (total/ours)   : {(sp_ms + topk_ms) / my_ms:.2f}x")
    print(f"{'='*60}\n")
