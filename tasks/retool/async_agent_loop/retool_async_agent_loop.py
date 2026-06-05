import asyncio

import hydra
import torch

try:
    import tasks.retool.agentic_rl  # noqa: F401 — register retool_dapo gem env
except ImportError:
    pass

from gpatch_v4.configs.utils import merge_hydra_config
from gpatch_v4.trainer import GrpoSingleCtrlTrainer


@hydra.main(config_path=".", config_name="test_config", version_base=None)
def main(cfg=None):
    config_cls = hydra.utils.get_class(cfg._target_)
    merged_obj = merge_hydra_config(config_cls, cfg)
    trainer = GrpoSingleCtrlTrainer()
    if merged_obj.checkpoint.convert_mcore_to_hf_offline:
        asyncio.run(trainer.conv_mcore_to_hf(merged_obj))
        return
    if merged_obj.debug.debug_engine_update_weight:
        asyncio.run(trainer.debug_update_weight(merged_obj))
        return
    asyncio.run(trainer.launch_then_run_with_recovery(merged_obj))


if __name__ == "__main__":
    main()
