from functools import partial
import time

import torch
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts, CheckpointPolicy
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func


def grad_fn(grad):
    return grad


def fn(q):
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
    out.register_hook(grad_fn)
    out2 = flash_attn_func(out, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
    out3 = out + out2
    return out3


def policy_fn(ctx, op, *args, **kwargs):
    # print(op.__name__)
    return CheckpointPolicy.MUST_RECOMPUTE


def test_fa2():
    q = torch.zeros(2, 64 * 1024, 16, 16, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    q = q + q

    # print('-' * 100)
    # print('fwd')
    torch.cuda.synchronize()
    t0 = time.time()

    out = q
    for i in range(2):
        if True:
            out = checkpoint(
                fn,
                out,
                use_reentrant=False,
                context_fn=partial(create_selective_checkpoint_contexts, policy_fn),
            )
        else:
            out = fn(out)
    loss = out.sum()

    torch.cuda.synchronize()
    t1 = time.time()
    # print(f'elapsed {t1-t0}')

    # print('-' * 100)
    # print('bwd')
    torch.cuda.synchronize()
    t0 = time.time()

    loss.backward()

    torch.cuda.synchronize()
    t1 = time.time()
    # print(f'elapsed {t1-t0}')
