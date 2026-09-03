import asyncio

import hydra
import torch

# isort: off
import gpatch_v4.core.device  # noqa: F401  # ensure device backend is initialized early
# isort: on

from gpatch_v4.configs.config import EmbeddingConfig
from gpatch_v4.configs.utils import merge_hydra_config
from gpatch_v4.trainer import FinetuneTrainer
from gpatch_v4.utils import log


@hydra.main(config_path="../configs/yaml", config_name="test_config", version_base=None)
def main(cfg: EmbeddingConfig):
    merged_obj = merge_hydra_config(EmbeddingConfig, cfg)

    trainer = FinetuneTrainer()
    if merged_obj.checkpoint.convert_mcore_to_hf_offline:
        asyncio.run(trainer.conv_mcore_to_hf(merged_obj))
        return
    asyncio.run(trainer.launch_then_run_with_recovery(merged_obj))


if __name__ == "__main__":
    main()
