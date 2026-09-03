import asyncio

from gpatch_v4.compat import ensure_typing_self

ensure_typing_self()

import hydra
import torch

from gpatch_v4.configs.config import FinetuneConfig
from gpatch_v4.configs.utils import merge_hydra_config
from gpatch_v4.trainer import FinetuneTrainer
from gpatch_v4.utils import log


@hydra.main(config_path="../configs/yaml", config_name="test_config", version_base=None)
def main(cfg: FinetuneConfig):
    """Entry point for LM supervised fine-tuning.

    Parameters
    ----------
    cfg : FinetuneConfig
        Hydra-resolved configuration.
    """
    merged_obj = merge_hydra_config(FinetuneConfig, cfg)

    trainer = FinetuneTrainer()
    if merged_obj.checkpoint.convert_mcore_to_hf_offline:
        asyncio.run(trainer.conv_mcore_to_hf(merged_obj))
        return
    asyncio.run(trainer.launch_then_run_with_recovery(merged_obj))


if __name__ == "__main__":
    main()
