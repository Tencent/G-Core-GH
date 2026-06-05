import asyncio
import collections
import os
import sys

import hydra
import torch

from gpatch_v4.configs.config import T2iRlConfig
from gpatch_v4.configs.utils import merge_hydra_config
from gpatch_v4.orches.custom_actor_registry import register_custom_actors
from gpatch_v4.trainer import T2iGrpoTrainer


@hydra.main(config_path="../configs/yaml", config_name="dit_rl_base_config", version_base=None)
def main(cfg: T2iRlConfig):
    """Entry point for T2I GRPO training.

    Parameters
    ----------
    cfg : T2iRlConfig
        Hydra-managed configuration object loaded from YAML.
    """
    merged_obj = merge_hydra_config(T2iRlConfig, cfg)
    register_custom_actors(merged_obj.training)
    trainer = T2iGrpoTrainer()
    asyncio.run(trainer.launch_then_run_with_recovery(merged_obj))


if __name__ == '__main__':
    main()
