from typing import Optional

import torch
import torch.distributed as dist

from megatron.core import mpu

from gpatch_v4.core.mappings import all_gather_from_context_parallel_region
from gpatch_v4.utils.training_utils import masked_mean


class _VocabParallelEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
        @torch.compile(dynamic=True)
        def mul_reduce(a, b):
            return (a * b).sum(dim=-1, keepdim=True)

        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(
            logits_max, op=dist.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group()
        )
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(normalized_sum_exp_logits, group=mpu.get_tensor_model_parallel_group())
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        sum_softmax_times_logits = mul_reduce(softmax_logits, vocab_parallel_logits)
        dist.all_reduce(sum_softmax_times_logits, group=mpu.get_tensor_model_parallel_group())
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        # reuse softmax_logits as grad
        vocab_parallel_logits.sub_(sum_softmax_times_logits)
        softmax_logits.mul_(vocab_parallel_logits)
        softmax_logits.mul_(grad_output.unsqueeze(dim=-1))
        # recover vocab_parallel_logits
        vocab_parallel_logits.add_(sum_softmax_times_logits)
        softmax_logits.mul_(-1)
        return softmax_logits


def vocab_parallel_entropy(
    vocab_parallel_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    ignore_cp: bool = False,
    pre_shifted: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token entropy from vocab-parallel logits.

    Parameters
    ----------
    pre_shifted : bool
        If True, the caller already applied the next-token shift to the data
        (e.g. Dynamic CP packed RL), so the trailing ``[:, :-1]`` truncation
        is skipped.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(scaled_entropy, per_token_entropy)`` where *scaled_entropy* is the
        (optionally masked) mean.
    """
    # input: vocab_parallel_logits [b, s, vocab_size // TP_SIZE]
    if mpu.get_context_parallel_world_size() > 1 and not ignore_cp:
        # entropy_unmasked [b, s]
        entropy_unmasked = _VocabParallelEntropy.apply(vocab_parallel_logits)
        per_token_entropy = all_gather_from_context_parallel_region(entropy_unmasked)
    else:
        per_token_entropy = _VocabParallelEntropy.apply(vocab_parallel_logits)

    if not pre_shifted:
        per_token_entropy = per_token_entropy[:, :-1]

    scaled_entropy = per_token_entropy.mean(
    ) if mask is None else masked_mean(per_token_entropy, mask)
    return scaled_entropy, per_token_entropy
