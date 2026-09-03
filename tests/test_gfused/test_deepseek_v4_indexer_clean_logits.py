import torch
from gpatch_v4.models.deepseek_v4.kernel.tilelang_indexer_fwd import (
    make_causal_cu_ks_and_cu_ke_for_bshd,
    clean_logits_,
    tl_indexer_fwd_impl,
)
device = "cuda:0"
S, H, D, m = 16384, 64, 128, 4
ctx = 524288
T = ctx // m
warmup, iters = 3, 10
q = torch.randn(S, H, D, device=device, dtype=torch.bfloat16)
k = torch.randn(T, D, device=device, dtype=torch.bfloat16)
w = torch.randn(S, H, device=device, dtype=torch.float32)
logits = torch.empty(S, T, device=device, dtype=torch.float32)
q_flat = q.view(S * H, D)
k_fwd = tl_indexer_fwd_impl(heads=H, index_dim=D)
k_clean = clean_logits_()
pos_h = torch.arange(S, device=device, dtype=torch.int32)
pos_t = pos_h + (ctx - S)
cu_ks_h, cu_ke_h = make_causal_cu_ks_and_cu_ke_for_bshd(S, T, m, device, positions=pos_h)
cu_ks_t, cu_ke_t = make_causal_cu_ks_and_cu_ke_for_bshd(S, T, m, device, positions=pos_t)
def time_it(fn):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st, ed = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        fn()
    ed.record()
    torch.cuda.synchronize()
    return st.elapsed_time(ed) / iters  # ms
def bench(tag, cu_ks, cu_ke):
    gemm_ms = time_it(lambda: k_fwd(q_flat, k, logits, w, cu_ks, cu_ke))
    logits.fill_(0)
    clean_ms = time_it(lambda: k_clean(logits, cu_ks, cu_ke))
    print(tag, f"gemm_ms={gemm_ms:.2f}", f"clean_ms={clean_ms:.2f}",
          f"clean/gemm={clean_ms / gemm_ms:.2f}")
bench("HEAD", cu_ks_h, cu_ke_h)
bench("TAIL", cu_ks_t, cu_ke_t)
