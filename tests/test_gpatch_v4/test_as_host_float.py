import torch

from gpatch_v4.utils.training_utils import get_scale_as_float, reduce_metrics


def test_get_scale_as_float_none():
    assert get_scale_as_float(None) is None


def test_get_scale_as_float_python_float():
    assert get_scale_as_float(1.25) == 1.25


def test_get_scale_as_float_cpu_tensor():
    out = get_scale_as_float(torch.tensor(3.5))
    assert isinstance(out, float)
    assert out == 3.5


def test_reduce_metrics_with_converted_grad_norm():
    # 复现 TE 返回 0-dim tensor 再经 get_scale_as_float 后 reduce_metrics 可跑通
    metrics = {
        "policy/grad_norm": [
            get_scale_as_float(torch.tensor(1.0)),
            get_scale_as_float(torch.tensor(3.0)),
        ],
    }
    out = reduce_metrics(metrics)
    assert out["policy/grad_norm"] == 2.0
