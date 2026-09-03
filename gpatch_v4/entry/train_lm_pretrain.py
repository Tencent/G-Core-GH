import asyncio

from gpatch_v4.compat import ensure_typing_self

ensure_typing_self()

import hydra

from gpatch_v4.configs.config import PretrainConfig
from gpatch_v4.configs.utils import merge_hydra_config
from gpatch_v4.trainer import FinetuneTrainer


@hydra.main(config_path="../configs/yaml", config_name="test_config", version_base=None)
def main(cfg: PretrainConfig):
    """Entry point for packed LM pretrain (lean PretrainActor path)."""
    merged_obj = merge_hydra_config(PretrainConfig, cfg)

    trainer = FinetuneTrainer()
    if merged_obj.checkpoint.convert_mcore_to_hf_offline:
        asyncio.run(trainer.conv_mcore_to_hf(merged_obj))
        return
    asyncio.run(trainer.launch_then_run_with_recovery(merged_obj))


if __name__ == "__main__":
    main()
