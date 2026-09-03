# 还有 diff 没对齐的 /home/xiaotaoliu/data/pretrainx/myaicoder/memory/dspark_implementation_analysis.md
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F

from gpatch_v4.models.deepseek_v4.dspark import DSparkBatch, DSparkForwardOutput


@dataclass
class DSparkLossResult:
    loss: torch.Tensor
    ce_loss: torch.Tensor
    l1_loss: torch.Tensor
    confidence_loss: torch.Tensor
    accept_rate_sums: torch.Tensor
    accept_rate_counts: torch.Tensor
    tau_sum: torch.Tensor
    block_count: torch.Tensor


def accumulate_dspark_report_metrics(
    totals: dict[str, torch.Tensor],
    result: DSparkLossResult,
    cp_group=None,
) -> None:
    loss_components = torch.stack(
        [
            result.loss.detach(),
            result.ce_loss.detach(),
            result.l1_loss.detach(),
            result.confidence_loss.detach(),
        ]
    )
    if cp_group is not None:
        dist.all_reduce(loss_components, group=cp_group)

    values = {
        "loss_components": loss_components,
        "accept_rate_sums": result.accept_rate_sums.detach(),
        "accept_rate_counts": result.accept_rate_counts.detach(),
        "tau_sum": result.tau_sum.detach(),
        "block_count": result.block_count.detach(),
    }
    for name, value in values.items():
        if name not in totals:
            totals[name] = torch.zeros_like(value)
        totals[name] += value


def dspark_loss_denominator(
    batch: DSparkBatch,
    *,
    block_size: int,
    loss_decay_gamma: float,
) -> torch.Tensor:
    """Count position-weighted supervision for one prepared microbatch.

    Parameters
    ----------
    batch : DSparkBatch
    block_size : int
    loss_decay_gamma : float

    Returns
    -------
    torch.Tensor
        Scalar denominator before cross-rank and cross-microbatch reduction.
    """
    positions = torch.arange(
        block_size,
        device=batch.eval_mask.device,
        dtype=torch.float32,
    )
    decay = torch.exp(-positions / loss_decay_gamma).view(1, 1, -1)
    return (batch.eval_mask.float() * decay).sum()


def calculate_dspark_loss(
    *,
    outputs: DSparkForwardOutput,
    global_denominator: torch.Tensor,
    dp_size: int,
    ce_loss_alpha: float,
    l1_loss_alpha: float,
    confidence_loss_alpha: float,
    loss_decay_gamma: float,
) -> DSparkLossResult:
    """Compute the three DSpark terms with FSDP2 step-global normalization.

    Parameters
    ----------
    outputs : DSparkForwardOutput
        Draft and frozen-target distributions for sampled anchor blocks.
    global_denominator : torch.Tensor
        Position-weighted valid-token count over every rank and microbatch.
    dp_size : int
    ce_loss_alpha : float
    l1_loss_alpha : float
    confidence_loss_alpha : float
    loss_decay_gamma : float

    Returns
    -------
    DSparkLossResult
        Backward loss plus unreduced metric numerators.
    """
    block_size = outputs.target_ids.shape[-1]
    positions = torch.arange(
        block_size,
        device=outputs.draft_logits.device,
        dtype=torch.float32,
    )
    decay = torch.exp(-positions / loss_decay_gamma).view(1, 1, -1)
    weights = outputs.eval_mask.float() * decay

    vocab_size = outputs.draft_logits.shape[-1]
    ce_per_token = F.cross_entropy(
        outputs.draft_logits.float().reshape(-1, vocab_size),
        outputs.target_ids.reshape(-1),
        reduction="none",
    ).reshape_as(outputs.target_ids)
    ce_num = (ce_per_token * weights).sum()

    draft_probs = torch.softmax(outputs.draft_logits.float(), dim=-1)
    target_probs = torch.softmax(outputs.target_logits.float(), dim=-1)
    l1_per_token = (draft_probs - target_probs).abs().sum(dim=-1)
    l1_num = (l1_per_token * weights).sum()
    accept_rate = (1.0 - 0.5 * l1_per_token).clamp(0.0, 1.0)
    confidence_per_token = F.binary_cross_entropy_with_logits(
        outputs.confidence_logits.float(),
        accept_rate.detach(),
        reduction="none",
    )
    confidence_num = (confidence_per_token * weights).sum()

    scale = float(dp_size) / global_denominator
    ce_loss = ce_num * scale
    l1_loss = l1_num * scale
    confidence_loss = confidence_num * scale
    loss = (
        ce_loss_alpha * ce_loss + l1_loss_alpha * l1_loss + confidence_loss_alpha * confidence_loss
    )

    with torch.no_grad():
        valid = outputs.eval_mask.float()
        accept_rate_sums = (accept_rate * valid).sum(dim=(0, 1))
        accept_rate_counts = valid.sum(dim=(0, 1))
        valid_blocks = outputs.block_keep_mask & outputs.eval_mask.any(dim=-1)
        tau = (accept_rate * valid).cumprod(dim=-1).sum(dim=-1) + 1.0
        tau_sum = (tau * valid_blocks.float()).sum()
        block_count = valid_blocks.sum()
    return DSparkLossResult(
        loss=loss,
        ce_loss=ce_loss,
        l1_loss=l1_loss,
        confidence_loss=confidence_loss,
        accept_rate_sums=accept_rate_sums,
        accept_rate_counts=accept_rate_counts,
        tau_sum=tau_sum,
        block_count=block_count,
    )


def reduce_dspark_metrics(
    metrics: dict[str, float],
    *,
    metric_prefix: str,
    block_size: int,
    device,
    loss: float,
    ce_loss: float,
    l1_loss: float,
    confidence_loss: float,
    accept_rate_sums: torch.Tensor,
    accept_rate_counts: torch.Tensor,
    tau_sum: torch.Tensor,
    block_count: torch.Tensor,
) -> None:
    """All-reduce DSpark step totals into ``metrics``.

    Loss scalars use AVG. Accept-rate and tau use SUM then divide so a rank
    with no valid tokens does not dilute the mean.

    Parameters
    ----------
    metrics : dict[str, float]
        Mutated in place with ``{metric_prefix}/dspark_*`` keys.
    metric_prefix : str
    block_size : int
        Length of the per-position accept-rate vectors.
    device
    loss : float
    ce_loss : float
    l1_loss : float
    confidence_loss : float
    accept_rate_sums : torch.Tensor
        Shape ``[block_size]``.
    accept_rate_counts : torch.Tensor
        Shape ``[block_size]``.
    tau_sum : torch.Tensor
    block_count : torch.Tensor
    """
    assert accept_rate_sums is not None
    assert accept_rate_counts is not None
    assert tau_sum is not None
    assert block_count is not None
    loss_t = torch.tensor(loss, device=device)
    ce_loss_t = torch.tensor(ce_loss, device=device)
    l1_loss_t = torch.tensor(l1_loss, device=device)
    confidence_loss_t = torch.tensor(confidence_loss, device=device)
    for value in (loss_t, ce_loss_t, l1_loss_t, confidence_loss_t):
        dist.all_reduce(value, op=dist.ReduceOp.AVG)
    metrics[f"{metric_prefix}/dspark_loss"] = loss_t.item()
    metrics[f"{metric_prefix}/dspark_ce_loss"] = ce_loss_t.item()
    metrics[f"{metric_prefix}/dspark_l1_loss"] = l1_loss_t.item()
    metrics[f"{metric_prefix}/dspark_confidence_loss"] = confidence_loss_t.item()
    dist.all_reduce(accept_rate_sums)
    dist.all_reduce(accept_rate_counts)
    dist.all_reduce(tau_sum)
    dist.all_reduce(block_count)
    for position in range(block_size):
        denominator = accept_rate_counts[position].clamp_min(1.0)
        metrics[f"{metric_prefix}/dspark_accept_rate_{position}"] = (
            accept_rate_sums[position] / denominator
        ).item()
    metrics[f"{metric_prefix}/dspark_tau"] = (tau_sum / block_count.clamp_min(1.0)).item()
