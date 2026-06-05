import asyncio

import hydra
import torch
from hydra.core.config_store import ConfigStore

from gpatch_v4.configs.config import InferenceConfig
from gpatch_v4.configs.utils import merge_hydra_config
from gpatch_v4.evaluate import InferenceRunner
from gpatch_v4.utils import log

cs = ConfigStore.instance()
cs.store(name="config_root", node=InferenceConfig)


@hydra.main(config_path="../configs/yaml", config_name="test_config", version_base=None)
def main(cfg: InferenceConfig):
    """Entry point for standalone inference.

    Parameters
    ----------
    cfg : InferenceConfig
        Hydra-resolved configuration.
    """
    merged_obj = merge_hydra_config(InferenceConfig, cfg)

    server = InferenceRunner()
    asyncio.run(server.start(merged_obj))


if __name__ == "__main__":
    main()
