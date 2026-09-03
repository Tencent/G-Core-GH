import pytest
import torch

from tile_kernels.quant import per_token_cast, per_token_cast_back

from gpatch_v4.models.deepseek_v4.kernel.tilelang_indexer_fwd import (
    clean_logits_,
    make_causal_cu_ks_and_cu_ke_for_bshd,
    tl_indexer_fwd_fp8_impl,
    tl_indexer_fwd_impl,
)

# DeepSeek-V4 Lightning Indexer (CSA) shapes from config:
#   index_n_heads=64, index_head_dim=128, compress_rate=4
INDEX_H = 64
INDEX_D = 128
COMPRESS_RATIO = 4

# seq_len: raw token length; seq_len_kv = seq_len // m. No CP — full causal 0..S-1.
MODEL_SHAPES = [
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
]


def calc_diff(x, y):
    # https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_deepgemm/example_deepgemm_fp8_2xAcc.py#L134-L138
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def report_diff(tag: str, shape, my_f: torch.Tensor, ref_f: torch.Tensor):
    abs_diff = (my_f - ref_f).abs()
    avg_abs = abs_diff.mean().item()
    avg_rel = (abs_diff / ref_f.abs().clamp_min(1e-8)).mean().item()
    diff = calc_diff(my_f, ref_f)
    print(
        f"  [{tag}] shape={shape} "
        f"avg_abs={avg_abs:.4e} avg_rel={avg_rel:.4e} calc_diff={diff:.4e}"
    )
    return avg_abs, avg_rel, diff


@pytest.mark.parametrize(
    "seq_lens",
    [
        (256,),
        (1024,),
        (8192,),
        (256, 512),
        (1024, 256, 768),
    ],
    ids=["s256", "s1024", "s8192", "packed-256-512", "packed-1024-256-768"],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_indexer_fwd_fp8_matches_fake_quant_bf16(seq_lens):
    torch.manual_seed(0)
    lq = sum(seq_lens)
    kv_lens = [seq_len // COMPRESS_RATIO for seq_len in seq_lens]
    lk = sum(kv_lens)
    heads, dim = 64, 128
    device = torch.device("cuda")
    fp8_blk_sz = 128
    scale_dtype = torch.float8_e8m0fnu

    q = torch.randn((lq * heads, dim), device=device, dtype=torch.bfloat16)
    k = torch.randn((lk, dim), device=device, dtype=torch.bfloat16)
    q_fp8, qs = per_token_cast(q, "e4m3", fp8_blk_sz, round_sf=True)
    k_fp8, ks = per_token_cast(k, "e4m3", fp8_blk_sz, round_sf=True)
    qs = qs.to(scale_dtype)
    ks = ks.to(scale_dtype)
    q_fake_quant = per_token_cast_back((q_fp8, qs.float()), "bf16", fp8_blk_sz)
    k_fake_quant = per_token_cast_back((k_fp8, ks.float()), "bf16", fp8_blk_sz)

    weights = torch.randn((lq, heads), device=device)
    cu_ks = torch.empty(lq, device=device, dtype=torch.int32)
    cu_ke = torch.empty(lq, device=device, dtype=torch.int32)
    q_start = 0
    k_start = 0
    for seq_len, kv_len in zip(seq_lens, kv_lens):
        q_end = q_start + seq_len
        positions = torch.arange(seq_len, device=device, dtype=torch.int32)
        cu_ks[q_start:q_end] = k_start
        cu_ke[q_start:q_end] = k_start + (positions + 1) // COMPRESS_RATIO
        q_start = q_end
        k_start += kv_len
    actual = torch.empty((lq, lk), device=device)
    expected = torch.empty_like(actual)

    fp8_kernel = tl_indexer_fwd_fp8_impl(
        heads=heads, index_dim=dim, fp8_block_size=fp8_blk_sz
    )
    bf16_kernel = tl_indexer_fwd_impl(
        heads=heads,
        index_dim=dim,
    )
    clean_kernel = clean_logits_()
    fp8_kernel(q_fp8, qs, k_fp8, ks, actual, weights, cu_ks, cu_ke)
    bf16_kernel(q_fake_quant, k_fake_quant, expected, weights, cu_ks, cu_ke)
    clean_kernel(actual, cu_ks, cu_ke)
    clean_kernel(expected, cu_ks, cu_ke)

    kv_positions = torch.arange(lk, device=device)
    valid = (kv_positions >= cu_ks[:, None]) & (kv_positions < cu_ke[:, None])
    avg_abs, avg_rel, diff = report_diff(
        "indexer", (seq_lens, lq, lk, heads, dim), actual[valid], expected[valid]
    )
    assert avg_abs < 1e-2, f"indexer avg_abs={avg_abs}"
    assert avg_rel < 1e-2, f"indexer avg_rel={avg_rel}"
    assert diff < 1e-5, f"indexer calc_diff={diff}"
    torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-2)


def _bench_cuda_ms(fn, warmup: int = 5, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _print_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_perf_indexer_fwd_fp8():
    # Full-sequence causal; bf16 vs quant+fp8 kernel.
    '''
    tests/test_gfused/test_deepseek_v4_indexer.py::test_perf_indexer_fwd_fp8
indexer fwd quant+fp8 vs bf16 (ms): H=64 D=128 m=4, causal=0..S-1
shape(S,T)    bf16     quant+fp8  fp8_vs_bf16
------------  -------  ---------  -----------
1024x256      0.057    0.083      0.69x
2048x512      0.157    0.161      0.97x
4096x1024     0.481    0.441      1.09x
8192x2048     1.616    1.406      1.15x
16384x4096    5.892    4.983      1.18x
32768x8192    22.430   18.768     1.20x
65536x16384   87.238   72.930     1.20x
131072x32768  344.388  287.513    1.20x
    '''
    device = torch.device("cuda")
    fp8_blk_sz = 128
    scale_dtype = torch.float8_e8m0fnu
    k_bf16 = tl_indexer_fwd_impl(heads=INDEX_H, index_dim=INDEX_D)
    k_fp8 = tl_indexer_fwd_fp8_impl(
        heads=INDEX_H, index_dim=INDEX_D, fp8_block_size=fp8_blk_sz, block_Q=128 // INDEX_H, block_N=256,
    )
    rows = []

    for seq_len in MODEL_SHAPES:
        assert seq_len % COMPRESS_RATIO == 0
        seq_len_kv = seq_len // COMPRESS_RATIO
        q = torch.randn(seq_len, INDEX_H, INDEX_D, device=device, dtype=torch.bfloat16)
        k = torch.randn(seq_len_kv, INDEX_D, device=device, dtype=torch.bfloat16)
        w = torch.randn(seq_len, INDEX_H, device=device, dtype=torch.float32)
        logits = torch.empty(seq_len, seq_len_kv, device=device, dtype=torch.float32)
        q_flat = q.view(seq_len * INDEX_H, INDEX_D)
        cu_ks, cu_ke = make_causal_cu_ks_and_cu_ke_for_bshd(
            seq_len, seq_len_kv, COMPRESS_RATIO, device
        )

        def run_bf16():
            k_bf16(q_flat, k, logits, w, cu_ks, cu_ke)

        def run_fp8():
            q_fp8, qs = per_token_cast(q_flat, "e4m3", fp8_blk_sz, round_sf=True)
            k_fp8_t, ks = per_token_cast(k, "e4m3", fp8_blk_sz, round_sf=True)
            qs = qs.to(scale_dtype)
            ks = ks.to(scale_dtype)
            k_fp8(q_fp8, qs, k_fp8_t, ks, logits, w, cu_ks, cu_ke)

        run_bf16()
        run_fp8()
        torch.cuda.synchronize()

        ms_bf16 = _bench_cuda_ms(run_bf16)
        ms_fp8 = _bench_cuda_ms(run_fp8)
        rows.append((
            f"{seq_len}x{seq_len_kv}",
            f"{ms_bf16:.3f}",
            f"{ms_fp8:.3f}",
            f"{ms_bf16 / ms_fp8:.2f}x",
        ))
        del q, k, w, logits, q_flat, cu_ks, cu_ke
        torch.cuda.empty_cache()

    print(
        f"\nindexer fwd quant+fp8 vs bf16 (ms): "
        f"H={INDEX_H} D={INDEX_D} m={COMPRESS_RATIO}, causal=0..S-1"
    )
    _print_table(["shape(S,T)", "bf16", "quant+fp8", "fp8_vs_bf16"], rows)
 