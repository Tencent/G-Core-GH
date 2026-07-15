"""opd_topk_logprobs_from_linear_ce 融合路径的 CPU 数值等价单测。

真正的 linear_cross_entropy 是 GPU Triton kernel，这里 mock 成纯 torch 参考实现
（full-logits + log_softmax），只校验融合函数新增的组合逻辑：以第 0 列作 label 拿
log-sum-exp 锚点、相对 logit gather、TP 掩码 + all-reduce、[S,B,K]↔[B,S,K] 布局，
以及 autograd 是否连通。TP=1 / CP=1 下应与「full logits 上 log_softmax 再 gather」逐元素一致，
并与非融合版 from_parallel_logits_to_opd_topk_logprobs 直接对拍一致。
"""
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

import gpatch_v4.utils.training_utils as tu


def _ref_logp_gather(hidden, weight, ids_bsk):
    # hidden [S,B,H], weight [V,H], ids_bsk [B,S,K] -> [B,S,K]
    logits = torch.einsum('sbh,vh->sbv', hidden.float(), weight.float())
    logp = F.log_softmax(logits, dim=-1).transpose(0, 1)  # [B,S,V]
    return torch.gather(logp, -1, ids_bsk)


def _fake_linear_cross_entropy(hidden, weight, labels, temperature, reduction, group,
                               return_entropy=False):
    # 复刻真实 kernel 返回值：NLL = -logp(label)，reshape 回 labels.shape
    hid = hidden.reshape(-1, hidden.shape[-1]).float()
    lab = labels.reshape(-1)
    logp = F.log_softmax(hid @ weight.float().t(), dim=-1)
    nll = -logp[torch.arange(lab.numel()), lab]
    return nll.view(labels.shape)


@contextmanager
def _mock_parallel():
    """把 TP/CP 折叠成单卡：分布式原语恒等，mpu 全返回 rank0/size1。"""
    with ExitStack() as es:
        p = es.enter_context
        p(patch.object(tu, "set_linear_ce_backend", lambda *a, **k: None))
        p(patch.object(tu, "linear_cross_entropy", _fake_linear_cross_entropy))
        p(patch.object(tu, "all_reduce_autograd", lambda t, group=None, **k: t))
        p(patch("torch.distributed.all_reduce", lambda *a, **k: None))
        p(patch.object(tu.mpu, "get_context_parallel_rank", lambda: 0))
        p(patch.object(tu.mpu, "get_context_parallel_world_size", lambda: 1))
        p(patch.object(tu.mpu, "get_tensor_model_parallel_rank", lambda: 0))
        p(patch.object(tu.mpu, "get_tensor_model_parallel_world_size", lambda: 1))
        p(patch.object(tu.mpu, "get_tensor_model_parallel_group", lambda: None))
        yield


def _run_fused(hidden, weight, ids, ignore_cp=True):
    output_layer = SimpleNamespace(tp_group=None, sequence_parallel=False, weight=None)
    lce_out = {"hidden_states": hidden, "weight": weight, "output_layer": output_layer}
    with _mock_parallel():
        return tu.opd_topk_logprobs_from_linear_ce(
            linear_ce_backend="split_n",
            linear_ce_output=lce_out,
            target_ids=ids,
            ignore_cp=ignore_cp,
        )


def _run_from_parallel(hidden, weight, ids, ignore_cp=True):
    # from_parallel 吃完整 logits [B,S,V]（TP=1 即整词表）
    logits_bsv = torch.einsum('sbh,vh->sbv', hidden.float(), weight.float()).transpose(0, 1)
    with _mock_parallel():
        return tu.from_parallel_logits_to_opd_topk_logprobs(
            vocab_parallel_logits=logits_bsv,
            target_ids=ids,
            ignore_cp=ignore_cp,
        )


def test_fused_matches_full_logits_gather():
    torch.manual_seed(0)
    S, B, H, V, K = 5, 2, 8, 20, 4
    hidden = torch.randn(S, B, H, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(V, H, dtype=torch.float32, requires_grad=True)
    ids = torch.randint(0, V, (B, S, K))

    out = _run_fused(hidden, weight, ids)
    ref = _ref_logp_gather(hidden, weight, ids)

    assert out.shape == (B, S, K)
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max()


def test_fused_matches_from_parallel_opd():
    # 直接与非融合版对拍：同一 hidden/weight/ids，输出逐元素一致。
    torch.manual_seed(3)
    S, B, H, V, K = 6, 3, 8, 24, 5
    hidden = torch.randn(S, B, H, dtype=torch.float32)
    weight = torch.randn(V, H, dtype=torch.float32)
    ids = torch.randint(0, V, (B, S, K))

    fused = _run_fused(hidden, weight, ids)
    baseline = _run_from_parallel(hidden, weight, ids)

    assert fused.shape == baseline.shape == (B, S, K)
    assert torch.allclose(fused, baseline, atol=1e-5), (fused - baseline).abs().max()


def test_fused_grad_flows():
    torch.manual_seed(1)
    S, B, H, V, K = 4, 3, 8, 16, 3
    hidden = torch.randn(S, B, H, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(V, H, dtype=torch.float32, requires_grad=True)
    ids = torch.randint(0, V, (B, S, K))

    _run_fused(hidden, weight, ids).sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()


def test_fused_grad_matches_from_parallel():
    # 梯度也对拍：两函数对同一输入的 d/d hidden 应一致。
    torch.manual_seed(4)
    S, B, H, V, K = 5, 2, 8, 20, 4
    ids = torch.randint(0, V, (B, S, K))
    weight_data = torch.randn(V, H, dtype=torch.float32)

    hidden_a = torch.randn(S, B, H, dtype=torch.float32, requires_grad=True)
    hidden_b = hidden_a.detach().clone().requires_grad_(True)
    weight_a = weight_data.clone().requires_grad_(True)
    weight_b = weight_data.clone().requires_grad_(True)

    _run_fused(hidden_a, weight_a, ids).sum().backward()
    _run_from_parallel(hidden_b, weight_b, ids).sum().backward()

    assert torch.allclose(hidden_a.grad, hidden_b.grad, atol=1e-5), \
        (hidden_a.grad - hidden_b.grad).abs().max()
    assert torch.allclose(weight_a.grad, weight_b.grad, atol=1e-5), \
        (weight_a.grad - weight_b.grad).abs().max()


def test_fused_col0_equals_logp_of_id0():
    # 第 0 列必须精确等于 logp(id_0)（差分项为 0）
    torch.manual_seed(2)
    S, B, H, V, K = 3, 2, 8, 12, 5
    hidden = torch.randn(S, B, H, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(V, H, dtype=torch.float32, requires_grad=True)
    ids = torch.randint(0, V, (B, S, K))

    out = _run_fused(hidden, weight, ids)
    ref = _ref_logp_gather(hidden, weight, ids)
    assert torch.allclose(out[..., 0], ref[..., 0], atol=1e-6)


if __name__ == "__main__":
    test_fused_matches_full_logits_gather()
    test_fused_matches_from_parallel_opd()
    test_fused_grad_flows()
    test_fused_grad_matches_from_parallel()
    test_fused_col0_equals_logp_of_id0()
    print("all passed")
