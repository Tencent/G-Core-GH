"""Moonlight Muon optimizer
adapted from https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py.
Param groups are split into two (use_muon=True / False)

Moonlight 论文: https://arxiv.org/abs/2502.16982

TODO(@rionawang): grad 做 svd 分解疑似错误，没有在 full tensor 上进行。
"""

import math

import torch


# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (b * A + c * A @ A)  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Equivalent to NVIDIA Emerging-Optimizers ``scale_mode="spectral" × extra_scale_factor=0.2``.
    """
    def __init__(
        self,
        muon_params,
        adamw_params=None,
        lr=1e-3,
        wd=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        extra_scale_factor=0.2,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    ):
        adamw_params = list(adamw_params) if adamw_params is not None else []

        param_groups = [
            {
                "params": list(muon_params),
                "use_muon": True,
                "lr": lr,
                "wd": wd,
                "momentum": momentum,
                "nesterov": nesterov,
                "ns_steps": ns_steps,
            },
            {
                "params": adamw_params,
                "use_muon": False,
                "lr": lr,
                "wd": wd,
                "adamw_betas": adamw_betas,
                "adamw_eps": adamw_eps,
            },
        ]
        super().__init__(param_groups, dict())
        self.extra_scale_factor = extra_scale_factor

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = self.extra_scale_factor * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["wd"]

            if group["use_muon"]:
                ############################
                #           Muon           #
                ############################
                momentum = group["momentum"]

                # generate weight updates
                for p in group["params"]:
                    # sanity check
                    g = p.grad
                    if g is None:
                        continue
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)

                    # calc update
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if group["nesterov"]:
                        g = g.add(buf, alpha=momentum)
                    else:
                        g = buf
                    u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                    if u.shape != p.shape:
                        u = u.view(p.shape)

                    # scale update
                    adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                    # apply weight decay
                    p.data.mul_(1 - lr * wd)

                    # apply update
                    p.data.add_(u, alpha=-adjusted_lr)
            else:
                ############################
                #       AdamW backup       #
                ############################
                beta1, beta2 = group["adamw_betas"]
                eps = group["adamw_eps"]

                for p in group["params"]:
                    g = p.grad
                    if g is None:
                        continue
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["moment1"] = torch.zeros_like(g)
                        state["moment2"] = torch.zeros_like(g)
                    state["step"] += 1
                    step = state["step"]
                    buf1 = state["moment1"]
                    buf2 = state["moment2"]
                    buf1.lerp_(g, 1 - beta1)
                    buf2.lerp_(g.square(), 1 - beta2)

                    g = buf1 / (eps + buf2.sqrt())

                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step
                    scale = bias_correction1 / bias_correction2**0.5
                    p.data.mul_(1 - lr * wd)
                    p.data.add_(g, alpha=-lr / scale)

        return loss
