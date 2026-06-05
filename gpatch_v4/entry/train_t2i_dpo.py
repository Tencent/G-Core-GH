import os

import hydra

from gpatch_v4.trainer.t2i_dpo_trainer import BaseDpoTrainer

config_path = os.environ.get("config_path", "./")
config_name = os.environ.get("config_name", "simple_fsdp2")


@hydra.main(config_path=config_path, config_name=config_name, version_base=None)
def main(cfg=None):
    # parse args
    import time
    time.sleep(2)
    trainer = hydra.utils.instantiate(cfg.trainer)
    assert isinstance(
        trainer, BaseDpoTrainer
    ), f"trainer_cls {type(trainer)} should be derived from BaseDpoTrainer"
    trainer.init()
    print("trainer.train_loop", flush=True)
    trainer.train_loop()
    trainer.finalize()
    time.sleep(2)


if __name__ == "__main__":
    main()
