from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from gpatch_v4.configs import DpoConfig, FinetuneConfig
from gpatch_v4.training_backend.fsdp2_backend.mtp_loss import calculate_mtp_loss
from gpatch_v4.training_backend.loss.registry import register_loss
from gpatch_v4.training_backend.loss_factory import compute_dpo_loss_core
from gpatch_v4.utils.training_utils import selective_log_softmax_raw


@dataclass
class Fsdp2FinetuneLossInput:
    labels_2d: torch.Tensor
    loss_mask_2d: torch.Tensor
    batch: Dict[str, Any]
    dp_size: int
    logits: Optional[torch.Tensor] = None
    use_linear_ce: bool = False
    per_token_linear_ce_loss: Optional[torch.Tensor] = None
    linear_ce_backend: Optional[str] = None
    global_n: Optional[torch.Tensor] = None
    vocab_size: Optional[int] = None
    loss_fct: Optional[nn.Module] = None
    enable_mtp: bool = False
    mtp_per_depth_h: Optional[list[torch.Tensor]] = None
    lm_head: Optional[nn.Module] = None
    cp_group: Any = None
    packed_seq_params: Any = None
    global_n_for_mtp: Optional[torch.Tensor] = None
    mtp_loss_scaling_factor: float = 0.1
    all_gather_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None


@dataclass
class Fsdp2FinetuneLossResult:
    loss: torch.Tensor
    main_loss: torch.Tensor
    mtp_loss: Optional[torch.Tensor] = None
    mtp_depth_losses: Optional[list[torch.Tensor]] = None
    mtp_depth_loss_metrics: Optional[list[torch.Tensor]] = None
    metrics: Dict[str, float] = field(default_factory=dict)


@register_loss(
    backends=("fsdp2", ),
    loss_name="cross_entropy",
    log_registration=True,
)
def fsdp2_cross_entropy_loss(
    config: FinetuneConfig,
    loss_input: Fsdp2FinetuneLossInput,
) -> Fsdp2FinetuneLossResult:
    labels_2d = loss_input.labels_2d
    loss_mask_2d = loss_input.loss_mask_2d
    labels = labels_2d.view(-1)
    loss_mask = loss_mask_2d.view(-1)
    dp_size = loss_input.dp_size
    global_n = loss_input.global_n
    assert global_n is not None

    if loss_input.use_linear_ce:
        per_token_linear_ce_loss = loss_input.per_token_linear_ce_loss
        assert per_token_linear_ce_loss is not None
        assert per_token_linear_ce_loss.shape == labels_2d.shape, (
            f"{per_token_linear_ce_loss.shape=} != {labels_2d.shape=}"
        )
        loss = per_token_linear_ce_loss.view(-1)
    else:
        assert loss_input.logits is not None
        assert loss_input.loss_fct is not None
        assert loss_input.vocab_size is not None
        logits = loss_input.logits.view(-1, loss_input.vocab_size)
        labels = labels.to(logits.device)
        loss = loss_input.loss_fct(logits, labels)
    loss = loss * loss_mask.to(loss.device)

    # Q: 为什么 `* dp_size`？
    # A: 因为在 dp rank 之间 reduce 缩小了尺度。
    # Q: 为什么不需要 `/ num_microbatches`？
    # A: 因为在 micro batch 之间 reduce 已经缩小了尺度。
    main_loss = torch.sum(loss) / global_n
    main_loss = main_loss * dp_size

    mtp_loss = None
    mtp_depth_losses = None
    mtp_depth_loss_metrics = None
    if loss_input.enable_mtp:
        assert loss_input.mtp_per_depth_h is not None, (
            "enable_mtp=True but model forward returned no mtp_per_depth_h"
        )
        assert loss_input.lm_head is not None
        assert loss_input.global_n_for_mtp is not None
        if loss_input.use_linear_ce:
            assert loss_input.linear_ce_backend is not None
        else:
            assert loss_input.loss_fct is not None
        mtp_depth_nums = calculate_mtp_loss(
            mtp_per_depth_h=loss_input.mtp_per_depth_h,
            labels=labels_2d.to(main_loss.device),
            lm_head=loss_input.lm_head,
            loss_fct=loss_input.loss_fct,
            loss_mask=loss_mask_2d.to(main_loss.device),
            cp_group=loss_input.cp_group,
            packed_seq_params=loss_input.packed_seq_params,
            use_linear_ce=loss_input.use_linear_ce,
            linear_ce_backend=loss_input.linear_ce_backend,
        )
        assert len(mtp_depth_nums) == loss_input.global_n_for_mtp.numel(
        ), (f"{len(mtp_depth_nums)=} != {loss_input.global_n_for_mtp.numel()=}")
        mtp_depth_losses = []
        mtp_depth_loss_metrics = []
        for depth, d_loss_local in enumerate(mtp_depth_nums):
            den = loss_input.global_n_for_mtp[depth]
            mtp_depth_losses.append(d_loss_local / den * dp_size)
            n_global_metric = d_loss_local.detach().clone()
            dist.all_reduce(n_global_metric)
            mtp_depth_loss_metrics.append(n_global_metric / den * dp_size)
        mtp_loss = torch.stack(mtp_depth_losses).sum(
        ) * (loss_input.mtp_loss_scaling_factor / max(len(mtp_depth_losses), 1))
        loss = main_loss + mtp_loss
    else:
        loss = main_loss

    return Fsdp2FinetuneLossResult(
        loss=loss,
        main_loss=main_loss,
        mtp_loss=mtp_loss,
        mtp_depth_losses=mtp_depth_losses,
        mtp_depth_loss_metrics=mtp_depth_loss_metrics,
    )


@register_loss(
    backends=("fsdp2", ),
    loss_name="dpo",
    log_registration=True,
)
def fsdp2_dpo_loss(
    config: DpoConfig,
    loss_input: Fsdp2FinetuneLossInput,
) -> Fsdp2FinetuneLossResult:
    assert loss_input.logits is not None
    assert loss_input.all_gather_fn is not None
    training_config = config.training
    labels_2d = loss_input.labels_2d
    batch = loss_input.batch
    dp_size = loss_input.dp_size

    safe_labels = labels_2d.clamp(min=0)
    policy_logps = selective_log_softmax_raw(loss_input.logits, safe_labels)
    policy_logps = loss_input.all_gather_fn(policy_logps)

    ref_logprobs = batch["ref_logprobs"]
    full_loss_mask = batch["full_loss_mask"].float()
    ref_len = ref_logprobs.shape[1]
    policy_logps = policy_logps[:, :ref_len]
    dpo_loss_mask = full_loss_mask[:, :ref_len].float()

    policy_seq_logps = (policy_logps * dpo_loss_mask).sum(-1)
    ref_seq_logps = (ref_logprobs.to(policy_logps.device) * dpo_loss_mask).sum(-1)

    B = policy_seq_logps.shape[0]
    assert B % 2 == 0, f"DPO microbatch must have even size, got {B}"
    rbs = B // 2
    policy_chosen_logps, policy_rejected_logps = policy_seq_logps.split(rbs)
    ref_chosen_logps, ref_rejected_logps = ref_seq_logps.split(rbs)

    losses, chosen_rewards, rejected_rewards = compute_dpo_loss_core(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=training_config.dpo_beta,
        label_smoothing=training_config.dpo_label_smoothing,
        loss_type=training_config.dpo_loss_type,
    )

    if training_config.dpo_ftx_gamma > 1e-6:
        chosen_mask_sum = dpo_loss_mask[:rbs].sum(-1).clamp_min(1.0)
        losses = losses - training_config.dpo_ftx_gamma * policy_chosen_logps / chosen_mask_sum

    main_loss = losses.sum() / training_config.train_gbs * dp_size

    with torch.no_grad():
        reward_acc = (chosen_rewards > rejected_rewards).float().mean()
        metrics = {
            "rewards-accuracies": reward_acc.item(),
            "rewards-chosen": chosen_rewards.mean().item(),
            "rewards-rejected": rejected_rewards.mean().item(),
            "rewards-margins": (chosen_rewards - rejected_rewards).mean().item(),
            "logps-chosen": policy_chosen_logps.mean().item(),
            "logps-rejected": policy_rejected_logps.mean().item(),
            "ref-logps-chosen": ref_chosen_logps.mean().item(),
            "ref-logps-rejected": ref_rejected_logps.mean().item(),
        }

    return Fsdp2FinetuneLossResult(
        loss=main_loss,
        main_loss=main_loss,
        metrics=metrics,
    )
