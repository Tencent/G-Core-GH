import torch

from gpatch_v4.training_backend.fsdp2_backend.dspark_loss import reduce_dspark_metrics


def test_reduce_dspark_metrics_writes_averaged_and_ratio_keys(monkeypatch):
    monkeypatch.setattr(
        "gpatch_v4.training_backend.fsdp2_backend.dspark_loss.dist.all_reduce",
        lambda *args, **kwargs: None,
    )
    metrics = {}
    reduce_dspark_metrics(
        metrics,
        metric_prefix="eval",
        block_size=2,
        device="cpu",
        loss=1.5,
        ce_loss=1.0,
        l1_loss=0.25,
        confidence_loss=0.25,
        accept_rate_sums=torch.tensor([1.0, 3.0]),
        accept_rate_counts=torch.tensor([2.0, 3.0]),
        tau_sum=torch.tensor(4.0),
        block_count=torch.tensor(2.0),
    )
    assert metrics["eval/dspark_loss"] == 1.5
    assert metrics["eval/dspark_ce_loss"] == 1.0
    assert metrics["eval/dspark_l1_loss"] == 0.25
    assert metrics["eval/dspark_confidence_loss"] == 0.25
    assert metrics["eval/dspark_accept_rate_0"] == 0.5
    assert metrics["eval/dspark_accept_rate_1"] == 1.0
    assert metrics["eval/dspark_tau"] == 2.0
