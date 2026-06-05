import asyncio
import collections
import os
import sys

import hydra
import torch

from gpatch_v4.configs.bagel_configs import BagelT2iRlConfig
from gpatch_v4.configs.utils import merge_hydra_config
from gpatch_v4.trainer import T2iGrpoTrainer


@hydra.main(config_path="../configs/yaml", config_name="dit_rl_base_config", version_base=None)
def main(cfg: BagelT2iRlConfig):
    merged_obj = merge_hydra_config(BagelT2iRlConfig, cfg)

    trainer = T2iGrpoTrainer()
    asyncio.run(trainer.launch_then_run_with_recovery(merged_obj))


if __name__ == '__main__':
    main()
