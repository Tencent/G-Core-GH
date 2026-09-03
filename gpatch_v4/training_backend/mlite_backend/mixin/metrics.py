from typing import Any, Dict

import torch
import torch.distributed as dist


class MetricsMixin:
    def _collect_metrics(
        self,
        runtime_metrics: Dict[str, Any],
        global_valid_tokens: torch.Tensor,
        max_seq_length: int,
        *,
        prefix: str,
    ) -> Dict[str, float]:
        is_output_rank = self.runtime.is_mp_src_rank_with_outputs(self.handle)
        metric_device = global_valid_tokens.device
        if is_output_rank:
            loss_sum = self._sum_runtime_metric(
                runtime_metrics,
                "_mlite_loss_sum",
                metric_device,
            )
            token_count = self._sum_runtime_metric(
                runtime_metrics,
                "_mlite_token_count",
                metric_device,
            )
        else:
            loss_sum = torch.zeros((), dtype=torch.float32, device=metric_device)
            token_count = torch.zeros((), dtype=torch.float32, device=metric_device)
        seq_length = torch.tensor(
            float(max_seq_length),
            dtype=torch.float32,
            device=metric_device,
        )
        # Training-token denominator lives on handle.metric_group (physical DP×CP
        # pool under dyn CP; otherwise the logical DP group). WORLD MAX then
        # publishes the pool result to PP/EP peers for logging.
        metric_group = getattr(self.handle, "metric_group", None)
        if dist.is_initialized():
            sum_pack = torch.stack([loss_sum, token_count])
            if metric_group is not None:
                dist.all_reduce(sum_pack, op=dist.ReduceOp.SUM, group=metric_group)
            else:
                dist.all_reduce(sum_pack, op=dist.ReduceOp.SUM)
            max_pack = torch.stack([sum_pack[0], sum_pack[1], seq_length])
            dist.all_reduce(max_pack, op=dist.ReduceOp.MAX)
            loss_sum, token_count, seq_length = max_pack[0], max_pack[1], max_pack[2]
        if token_count.item() <= 0:
            raise RuntimeError("mlite runtime returned no valid-token metrics")
        if not torch.allclose(token_count, global_valid_tokens):
            raise RuntimeError(
                "mlite runtime token count does not match the packed-batch token count"
            )
        return {
            f"{prefix}/lm_loss": float((loss_sum / token_count).item()),
            f"{prefix}/seq_length": float(seq_length.item()),
        }

    @staticmethod
    def _sum_runtime_metric(
        runtime_metrics: Dict[str, Any],
        key: str,
        device: torch.device,
    ) -> torch.Tensor:
        values = runtime_metrics.get(key)
        if not values:
            raise RuntimeError(f"mlite runtime did not return metric {key!r}")
        if not isinstance(values, list):
            values = [values]
        total = torch.zeros((), dtype=torch.float32, device=device)
        for value in values:
            total += torch.as_tensor(value, dtype=torch.float32, device=device)
        return total
