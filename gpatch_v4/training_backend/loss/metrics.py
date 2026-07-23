from typing import Any, Dict

import torch

from gpatch_v4.training_backend.loss.registry import register_histogram_finalizer

_STEER_HISTOGRAM_BIN_COUNT = 256
_STEER_ENTROPY_CHANGE_LOG_MAX = 10.0
_STEER_HISTOGRAM_QUANTILES = (0.5, 0.9, 0.99)


def _steer_histogram(
    values: torch.Tensor,
    value_min: float,
    value_max: float,
    include_overflow: bool,
) -> torch.Tensor:
    scaled = (values - value_min) * _STEER_HISTOGRAM_BIN_COUNT / (value_max - value_min)
    bin_indices = scaled.to(torch.long).clamp(min=0, max=_STEER_HISTOGRAM_BIN_COUNT - 1)
    if include_overflow:
        bin_indices = torch.where(
            values > value_max,
            torch.full_like(bin_indices, _STEER_HISTOGRAM_BIN_COUNT),
            bin_indices,
        )
    return torch.bincount(
        bin_indices,
        minlength=_STEER_HISTOGRAM_BIN_COUNT + int(include_overflow),
    ).float()


def steer_histogram_quantiles(
    histogram: torch.Tensor,
    value_min: float,
    value_max: float,
    log_space: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_count = histogram.sum()
    if total_count == 0:
        return tuple(torch.zeros((), device=histogram.device) for _ in _STEER_HISTOGRAM_QUANTILES)

    cdf = histogram.cumsum(dim=0)
    thresholds = total_count * torch.tensor(
        _STEER_HISTOGRAM_QUANTILES, dtype=histogram.dtype, device=histogram.device
    )
    bin_indices = torch.searchsorted(cdf, thresholds).clamp(max=_STEER_HISTOGRAM_BIN_COUNT - 1)
    bin_width = (value_max - value_min) / _STEER_HISTOGRAM_BIN_COUNT
    quantiles = value_min + (bin_indices.to(histogram.dtype) + 0.5) * bin_width
    if log_space:
        quantiles = torch.expm1(quantiles)
    return tuple(quantiles.unbind())


def compute_steer_histograms(
    valid_token_weights: torch.Tensor,
    valid_entropy_change_metric: torch.Tensor,
    token_weight_min: float,
) -> Dict[str, torch.Tensor]:
    return {
        "steer/token_weight_histogram":
            _steer_histogram(
                valid_token_weights,
                value_min=token_weight_min,
                value_max=1.0,
                include_overflow=False,
            ),
        "steer/entropy_change_metric_histogram":
            _steer_histogram(
                torch.log1p(valid_entropy_change_metric),
                value_min=0.0,
                value_max=_STEER_ENTROPY_CHANGE_LOG_MAX,
                include_overflow=True,
            ),
    }


@register_histogram_finalizer("steer")
def finalize_steer_histogram_metrics(
    config: Any,
    histograms: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    metrics = {}
    token_weight_histogram = histograms.get("steer/token_weight_histogram")
    if token_weight_histogram is not None:
        p50, p90, p99 = steer_histogram_quantiles(
            token_weight_histogram,
            value_min=config.ppo.steer_token_weight_min,
            value_max=1.0,
            log_space=False,
        )
        metrics.update(
            {
                "steer/token_weight_p50": p50,
                "steer/token_weight_p90": p90,
                "steer/token_weight_p99": p99,
            }
        )

    entropy_change_histogram = histograms.get("steer/entropy_change_metric_histogram")
    if entropy_change_histogram is not None:
        p50, p90, p99 = steer_histogram_quantiles(
            entropy_change_histogram,
            value_min=0.0,
            value_max=_STEER_ENTROPY_CHANGE_LOG_MAX,
            log_space=True,
        )
        metrics.update(
            {
                "steer/entropy_change_metric_p50":
                    p50,
                "steer/entropy_change_metric_p90":
                    p90,
                "steer/entropy_change_metric_p99":
                    p99,
                "steer/entropy_change_metric_histogram_overflow_frac":
                    entropy_change_histogram[-1] / entropy_change_histogram.sum().clamp(min=1),
            }
        )
    return metrics
