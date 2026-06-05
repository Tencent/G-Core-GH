from typing import Dict, List, Union

import numpy as np
import torch
from tensordict import TensorDict


def union_two_dict(dict1: Dict, dict2: Dict):
    """Union two dicts. Raises if a shared key has differing values.

    Args:
        dict1:
        dict2:

    Returns:
    """
    for key, val in dict2.items():
        if key in dict1:
            assert dict2[key] == dict1[
                key], f"{key} in meta_dict1 and meta_dict2 are not the same object"
        dict1[key] = val

    return dict1


def divide_by_chunk_size(data: Union[np.ndarray, TensorDict],
                         chunk_sizes: List[int]) -> List[Union[np.ndarray, TensorDict]]:
    """将 numpy 数组按 chunks 大小切分。"""
    if not isinstance(data, (np.ndarray, TensorDict)):
        raise TypeError("Input 'array' must be a numpy ndarray or a TensorDict.")

    if not all(isinstance(size, int) and size > 0 for size in chunk_sizes):
        raise ValueError("All chunk sizes must be positive integers.")

    total_size = sum(chunk_sizes)
    if total_size != len(data):
        raise ValueError(
            f"The sum of chunk_sizes ({total_size}) does not match the size of the array ({len(data)})."
        )

    split_data = []
    start_index = 0
    for size in chunk_sizes:
        end_index = start_index + size
        split_data.append(data[start_index:end_index])
        start_index = end_index
    return split_data


def append_to_dict(data: Dict, new_data: Dict):
    for key, val in new_data.items():
        if key not in data:
            data[key] = []
        data[key].append(val)


def pad_to_length(tensor: torch.Tensor, length, pad_value, dim=-1):
    if tensor.size(dim) >= length:
        indices = [slice(None)] * tensor.ndim
        indices[dim] = slice(0, length)
        return tensor[indices]
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)],
            dim=dim
        )


def aggregate_metrics(history_metrics: List[Dict], metrics_agg_mode: Dict[str,
                                                                          str]) -> Dict[str, float]:
    """Aggregate metrics from history based on the specified modes.

    Args:
        history_metrics: Per-step metric dicts.
        metrics_agg_mode: Mode per metric name.
                         Supported: "sum", "mean", "min", "max", "last", "first".

    Returns:
        Aggregated metrics.
    """
    # Collect all metrics from history
    all_metrics = {}
    for metrics in history_metrics:
        for k, v in metrics.items():
            if k not in all_metrics:
                all_metrics[k] = []
            all_metrics[k].append(float(v))

    # Aggregate metrics based on mode
    aggregated_metrics = {}
    for metric_name, values in all_metrics.items():
        mode = metrics_agg_mode.get(metric_name, "mean")  # default to mean
        if mode == "sum":
            aggregated_metrics[metric_name] = float(np.sum(values))
        elif mode == "mean":
            aggregated_metrics[metric_name] = float(np.mean(values))
        elif mode == "min":
            aggregated_metrics[metric_name] = float(np.min(values))
        elif mode == "max":
            aggregated_metrics[metric_name] = float(np.max(values))
        elif mode == "last":
            aggregated_metrics[metric_name] = float(values[-1])
        elif mode == "first":
            aggregated_metrics[metric_name] = float(values[0])
        else:
            # Default to mean for unknown modes
            aggregated_metrics[metric_name] = float(np.mean(values))

    return aggregated_metrics
