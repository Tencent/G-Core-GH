import dataclasses
import functools
import gc
import os
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from time import time
from typing import Callable, Optional

import hydra
import torch

from gpatch_v4.trainer.bagel_trainer import BagelTrainer
from gpatch_v4.utils import import_fn_from_path

bagel_config_path = os.environ.get("bagel_config_path", "./")
bagel_config_name = os.environ.get("bagel_config_name", "bagel_finetune_hydra")


@hydra.main(config_path=bagel_config_path, config_name=bagel_config_name, version_base=None)
def main(cfg=None):
    # 使用 Hydra instantiate 直接创建 trainer
    # cfg.trainer 包含 _target_ 和所有参数
    trainer = hydra.utils.instantiate(cfg.trainer)

    assert isinstance(
        trainer, BagelTrainer
    ), f"trainer {type(trainer)} should be derived from BagelTrainer"
    trainer.init_distributed()
    trainer.init_log()
    trainer.build_model_and_optimizer()
    trainer.train_loop()
    trainer.finalize()


if __name__ == "__main__":
    main()
