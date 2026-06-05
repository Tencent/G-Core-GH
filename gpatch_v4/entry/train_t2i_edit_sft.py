import asyncio
import collections
import os
import sys

import hydra
import torch

from gpatch_v4.configs.config import T2iEditSftConfig
from gpatch_v4.configs.utils import merge_hydra_config
from gpatch_v4.orches.custom_actor_registry import register_custom_actors
from gpatch_v4.trainer import T2iEditSftTrainer


@hydra.main(
    config_path="../configs/yaml", config_name="t2i_edit_sft_base_config", version_base=None
)
def main(cfg: T2iEditSftConfig):
    merged_obj = merge_hydra_config(T2iEditSftConfig, cfg)
    trainer = T2iEditSftTrainer()
    asyncio.run(trainer.launch_then_run_with_recovery(merged_obj))


if __name__ == '__main__':
    main()
