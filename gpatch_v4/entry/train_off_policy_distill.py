import asyncio

import hydra
import torch

from gpatch_v4.configs.config import OffPolicyDistillConfig
from gpatch_v4.configs.utils import merge_hydra_config
from gpatch_v4.trainer import OffPolicyDistillTrainer
from gpatch_v4.transfer import close_tq_connector


@hydra.main(config_path="../configs/yaml", config_name="test_config", version_base=None)
def main(cfg: OffPolicyDistillConfig):
    """Entry point for off-policy distillation training.

    Parameters
    ----------
    cfg : OffPolicyDistillConfig
        Hydra-resolved configuration.
    """
    merged_obj = merge_hydra_config(OffPolicyDistillConfig, cfg)

    trainer = OffPolicyDistillTrainer()
    if merged_obj.checkpoint.convert_mcore_to_hf_offline:
        asyncio.run(trainer.conv_mcore_to_hf(merged_obj))
        return
    try:
        if merged_obj.debug.test_ray_rpc:
            asyncio.run(trainer.test_ray_rpc(merged_obj))
            return
        asyncio.run(trainer.launch_then_run_with_recovery(merged_obj))
    finally:
        close_tq_connector()


if __name__ == "__main__":
    main()
