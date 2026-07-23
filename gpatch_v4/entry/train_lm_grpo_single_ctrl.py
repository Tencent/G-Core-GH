import asyncio

import hydra
from omegaconf import OmegaConf, open_dict

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.configs.utils import merge_hydra_config
from gpatch_v4.trainer import GrpoSingleCtrlTrainer
from gpatch_v4.utils.placement import (
    is_partial_colocated,
    validate_partial_colocated_config,
)


@hydra.main(config_path="../configs/yaml", config_name="test_config", version_base=None)
def main(cfg: RlConfig):
    """Entry point for single-controller LM GRPO training.

    Supports both colocate and disaggregated placement types.
    Both modes use a driver-side loop (single_controller=True).

    Parameters
    ----------
    cfg : RlConfig
        Hydra-resolved configuration.
    """
    merged_obj = merge_hydra_config(RlConfig, cfg)

    assert merged_obj.training.single_controller, (
        "this entry point requires training.single_controller=True"
    )

    is_colocate = merged_obj.placement_type == "colocate"
    is_disaggregated = merged_obj.placement_type == "disaggregated"
    is_partial = is_partial_colocated(merged_obj)
    assert is_colocate or is_disaggregated or is_partial, (
        f"unsupported placement_type={merged_obj.placement_type}"
    )
    if is_colocate:
        assert not merged_obj.training.async_rollout, (
            "colocate requires training.async_rollout=False"
        )
        assert merged_obj.training.rollout_max_staleness == 0, (
            "colocate requires training.rollout_max_staleness=0"
        )
    elif is_partial:
        validate_partial_colocated_config(merged_obj)
    else:
        assert merged_obj.training.async_rollout, (
            "disaggregated requires training.async_rollout=True"
        )
        assert merged_obj.training.rollout_max_staleness >= 0, (
            "training.rollout_max_staleness must be >= 0"
        )

    trainer = GrpoSingleCtrlTrainer()
    if merged_obj.debug.debug_engine_update_weight:
        asyncio.run(trainer.debug_update_weight(merged_obj))
        return
    asyncio.run(trainer.launch_then_run_with_recovery(merged_obj))


if __name__ == "__main__":
    main()
