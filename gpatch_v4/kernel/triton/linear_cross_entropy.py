# adapted from https://github.com/verl-project/verl/blob/main/verl/utils/kernel/linear_cross_entropy.py

import typing

import torch
import torch.distributed as dist

from . import linear_ce_kernels as kernels

_BACKEND_MAP = {
    "fuse_mn": kernels.BackwardEnum._Total_Fuse_MN,
    "separate": kernels.BackwardEnum._Total_Separate,
    "split_n": kernels.BackwardEnum._Split_Dlogits_N,
}


def set_linear_ce_backend(backend: str):
    """Set the backward method for linear cross entropy.

    Parameters
    ----------
    backend : str
        One of ``"fuse_mn"``, ``"separate"``, ``"split_n"``.

        - ``"fuse_mn"``: Fuse d_logits/d_hidden/d_weight into single kernel,
          no intermediate d_logits buffer. Requires fp32 for d_hidden/d_weight.
          Best memory, best performance.
        - ``"separate"``: Compute full d_logits buffer, then separate
          matmul for d_hidden and d_weight. Uses native dtype.
        - ``"split_n"``: Split d_logits along vocab dimension, loop over
          splits. Balanced memory/performance. Default.
    """
    if backend not in _BACKEND_MAP:
        raise ValueError(
            f"Unknown linear_ce_backend: '{backend}'. "
            f"Choose from: {sorted(_BACKEND_MAP.keys())}"
        )
    kernels.set_backward_method(_BACKEND_MAP[backend])


class LinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        temperature: typing.Optional[float] = 1.0,
        reduction: typing.Optional[str] = "none",
        dist_process_group: typing.Optional[dist.ProcessGroup] = None,
    ) -> list[torch.Tensor]:
        """_summary_

        Args:
            ctx (_type_): _description_
            hidden (torch.Tensor): (batch_size, num_tokens, hidden_size) -> (batch_size * num_tokens, hidden_size)
            weight (torch.Tensor): (vocab_size, hidden_size)
            labels (torch.Tensor): (batch_size, num_tokens) -> (batch_size * num_tokens, )
            temperature (typing.Optional[float], optional): _description_. Defaults to 1.0.
            reduction (typing.Optional[str], optional): _description_. Defaults to "none".
            dist_process_group (typing.Optional[dist.ProcessGroup], optional): _description_. Defaults to None.

        Returns:
            typing.List[torch.Tensor]: _description_
        """

        assert isinstance(
            temperature, float
        ), f"temperature must be a float, but got {type(temperature)}"
        assert isinstance(reduction, str), f"reduction must be a str, but got {type(reduction)}"
        with torch.cuda.nvtx.range("LinearCrossEntropy-forward"):
            REDUCTION = kernels.get_entropy_reduction_enum_number(reduction.lower())

            original_hidden_shape = hidden.shape
            if len(hidden.shape) != 2:
                hidden = hidden.view(-1, hidden.shape[-1])  # (batch_size * num_tokens, hidden_size)
            if len(labels.shape) != 1:
                labels = labels.view(-1)

            logprobs, entropy, _maximum, _accumulate, _entropy_b = kernels.efficient_entropy_forward(
                hidden, weight, labels, REDUCTION, temperature, dist_process_group
            )

            ctx.save_for_backward(hidden, weight, labels, _maximum, _accumulate, _entropy_b)
            ctx.original_hidden_shape = original_hidden_shape
            ctx.REDUCTION = REDUCTION
            ctx.dist_process_group = dist_process_group
            ctx.should_return_fp32_grad = False
            ctx.temperature = temperature
        return logprobs, entropy

    @staticmethod
    def backward(ctx, dlogprobs: torch.Tensor, dentropy: torch.Tensor) -> list[torch.Tensor]:
        with torch.cuda.nvtx.range("LinearCrossEntropy-backward"):
            (hidden, weight, labels, _maximum, _accumulate, _entropy_b) = ctx.saved_tensors
            REDUCTION = ctx.REDUCTION
            dist_process_group = ctx.dist_process_group
            should_return_fp32_grad = ctx.should_return_fp32_grad
            temperature = ctx.temperature

            d_hidden, d_weight = kernels.efficient_entropy_backward(
                dlogprobs,
                dentropy,
                hidden,
                weight,
                labels,
                _maximum,
                _accumulate,
                _entropy_b,
                REDUCTION,
                should_return_fp32_grad,
                temperature,
                dist_process_group,
            )
            d_hidden = d_hidden.view(ctx.original_hidden_shape)

        return (d_hidden, d_weight, None, None, None, None)


def linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: typing.Optional[float] = 1.0,
    reduction: typing.Optional[str] = "none",
    dist_process_group: typing.Optional[dist.ProcessGroup] = None,
    return_entropy: bool = False,
):
    loss, entropy = LinearCrossEntropy.apply(
        hidden,
        weight,
        labels,
        temperature,
        reduction,
        dist_process_group,
    )
    # linear_cross_entropy 返回 log P(label)（负值），取负得到 NLL（正值），
    # 与 vocab_parallel_cross_entropy 返回值语义一致
    loss = (-loss).view(labels.shape)

    if return_entropy:
        entropy = entropy.view(labels.shape)
        return loss, entropy
    else:
        return loss
