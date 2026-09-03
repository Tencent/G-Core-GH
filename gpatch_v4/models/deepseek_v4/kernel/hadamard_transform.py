import torch

try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    hadamard_transform = None


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dim by the normalized Walsh-Hadamard matrix.

    NOTE: dsv4 官方实现中写的 "randomized Hadamard" 是错误的，这是确定性的 Hadamard Transform.
    不要“顺手补上”随机符号矩阵，否则 Q、K 和不同 CP rank 使用不一致的 Hadamard 矩阵会破坏数学等价性。
    """
    assert hadamard_transform is not None, "fast_hadamard_transform is not installed"
    d = x.size(-1)
    assert d > 0 and d & (d - 1) == 0, f"Hidden size must be a power of 2, got {d}"
    assert x.dtype == torch.bfloat16, f"Expected bfloat16, got {x.dtype}"
    return hadamard_transform(x, scale=d**-0.5)
