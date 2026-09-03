from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist

from megatron.core import mpu


class DynamicBatchMcoreMixin:
    """MCore overrides for dynamic train-step batches."""
    def _compute_step_gbs_and_token_cnt(
        self,
        batch: List[Dict[str, Any]],
    ) -> Tuple[int, int, torch.Tensor]:
        assert batch
        device = torch.cuda.current_device()
        local_stats = torch.zeros(3, device=device, dtype=torch.float32)
        local_stats[0] = len(batch)
        if "sample_mask" in batch[0]:
            assert all("sample_mask" in sample for sample in batch)
            local_stats[1] = torch.stack([sample["sample_mask"] for sample in batch]).float().sum()
        else:
            local_stats[1] = len(batch)
        local_stats[2] = sum(sample["mask"].float().sum() for sample in batch)
        dist.all_reduce(local_stats, group=mpu.get_data_parallel_group())
        return int(local_stats[0].item()), int(local_stats[1].item()), local_stats[2]

    def _static_num_microbatches(self, batch: List[Dict[str, Any]]) -> int:
        local_count = len(batch)
        counts = [None] * mpu.get_data_parallel_world_size()
        dist.all_gather_object(counts, local_count, group=mpu.get_data_parallel_group())
        assert len(set(counts)) == 1, (
            f"static dynamic-batch training requires equal local sample counts, got {counts}; "
            "enable dynamic CP"
        )
        assert local_count % self.training_config.train_mbs == 0, (
            f"local sample count {local_count} is not divisible by "
            f"train_mbs {self.training_config.train_mbs}; enable dynamic CP"
        )
        return local_count // self.training_config.train_mbs

    @staticmethod
    def _assert_equal_num_microbatches(num_microbatches: int) -> None:
        counts = [None] * mpu.get_data_parallel_world_size()
        dist.all_gather_object(
            counts,
            num_microbatches,
            group=mpu.get_data_parallel_group(),
        )
        assert len(set(counts)) == 1, f"DP ranks have different num_microbatches: {counts}"
