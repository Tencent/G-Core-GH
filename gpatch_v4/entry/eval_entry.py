import asyncio

import hydra
import torch

from gpatch_v4.configs.config import EvaluateConfig
from gpatch_v4.configs.utils import merge_hydra_config
from gpatch_v4.evaluate import EvaluateRunner
from gpatch_v4.utils import log


@hydra.main(config_path="../configs/yaml", config_name="test_config", version_base=None)
def main(cfg: EvaluateConfig):
    """Entry point for model evaluation.

    Parameters
    ----------
    cfg : EvaluateConfig
        Hydra-resolved configuration.
    """
    merged_obj = merge_hydra_config(EvaluateConfig, cfg)

    trainer = EvaluateRunner()
    asyncio.run(trainer.evaluate(merged_obj))


if __name__ == "__main__":
    main()
