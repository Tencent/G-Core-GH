import asyncio

import hydra

from gpatch_v4.configs.config import RewardConfig
from gpatch_v4.configs.utils import merge_hydra_config
from gpatch_v4.trainer import RewardTrainer


@hydra.main(config_path="../configs/yaml", config_name="test_config", version_base=None)
def main(cfg: RewardConfig):
    """Entry point for Bradley-Terry reward-model training.

    Parameters
    ----------
    cfg : RewardConfig
        Hydra-resolved configuration.
    """
    merged_obj = merge_hydra_config(RewardConfig, cfg)

    trainer = RewardTrainer()
    if merged_obj.checkpoint.convert_mcore_to_hf_offline:
        asyncio.run(trainer.conv_mcore_to_hf(merged_obj))
        return
    asyncio.run(trainer.launch_then_run_with_recovery(merged_obj))


if __name__ == "__main__":
    main()
