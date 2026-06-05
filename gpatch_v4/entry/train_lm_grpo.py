import asyncio

import hydra
import torch

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.configs.utils import merge_hydra_config
from gpatch_v4.trainer import GrpoTrainer


@hydra.main(config_path="../configs/yaml", config_name="test_config", version_base=None)
def main(cfg: RlConfig):
    """Entry point for LM GRPO training.

    Parameters
    ----------
    cfg : RlConfig
        Hydra-resolved configuration.
    """
    merged_obj = merge_hydra_config(RlConfig, cfg)

    trainer = GrpoTrainer()
    if merged_obj.checkpoint.convert_mcore_to_hf_offline:
        asyncio.run(trainer.conv_mcore_to_hf(merged_obj))
        return
    if merged_obj.debug.debug_engine_update_weight:
        asyncio.run(trainer.debug_update_weight(merged_obj))
        return
    asyncio.run(trainer.launch_then_run_with_recovery(merged_obj))


if __name__ == "__main__":
    main()
