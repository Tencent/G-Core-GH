import logging
import os

from gpatch_v4.utils.common_utils import log, reorder_dict_keys_by_prefix


class TrainReporterSingleton:
    """Singleton managing TensorBoard and WandB writers for training metrics."""
    tb_writer = None
    wandb_writer = None
    verl_metric_map = None

    @classmethod
    def get_tensorboard_writer(cls):
        return cls.tb_writer

    @classmethod
    def get_wandb_writer(cls):
        return cls.wandb_writer

    @classmethod
    def _set_tensorboard_writer(cls, report_config):
        cls._check_writer_not_initialized("tensorboard")
        tensorboard_dir = getattr(report_config, 'tensorboard_dir', None)
        if not tensorboard_dir:
            return

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            print("TensorBoard not available. Skipping initialization.")
            return

        cls.tb_writer = SummaryWriter(
            log_dir=tensorboard_dir, max_queue=getattr(report_config, 'tensorboard_queue_size', 10)
        )

    @classmethod
    def _set_wandb_writer(cls, report_config, config):
        cls._check_writer_not_initialized("wandb")
        if not getattr(report_config, 'wandb_project', None):
            return

        import wandb
        save_dir = _ensure_dir(report_config.wandb_save_dir)
        assert report_config.wandb_key is not None, "WandB key must be provided."
        assert report_config.wandb_host is not None, "WandB host must be provided."
        wandb.login(key=report_config.wandb_key, host=report_config.wandb_host)
        wandb.init(
            dir=save_dir,
            name=report_config.wandb_exp_name,
            project=report_config.wandb_project,
            id=report_config.wandb_run_id,
            config=vars(config)
        )
        cls.wandb_writer = wandb
        log(
            f"wandb initialized. project {report_config.wandb_project} "
            f"name {report_config.wandb_exp_name} "
            f"run_url {wandb.run.get_url() if wandb.run else 'N/A'}"
        )

    @classmethod
    def _check_writer_not_initialized(cls, writer_type):
        if writer_type == "tensorboard" and cls.tb_writer is not None:
            raise RuntimeError("TensorBoard writer already initialized.")
        if writer_type == "wandb" and cls.wandb_writer is not None:
            raise RuntimeError("WandB writer already initialized.")

    @classmethod
    def _translate_to_verl(cls, metrics: dict) -> dict:
        """Translate gcore metrics into verl-named metrics per ``verl_metric_map``.

        Parameters
        ----------
        metrics : dict
            Current gcore metrics for this step.

        Returns
        -------
        dict
            Newly produced ``{verl_key: value}`` entries. gcore keys absent from
            this step or mapped to ``None`` are skipped; a list target fans the
            same value out to multiple verl keys.
        """
        translated = {}
        for gcore_key, verl_target in cls.verl_metric_map.items():
            if verl_target is None or gcore_key not in metrics:
                continue
            targets = verl_target if isinstance(verl_target, list) else [verl_target]
            for verl_key in targets:
                translated[verl_key] = metrics[gcore_key]
        return translated

    @classmethod
    def log_and_report(cls, metrics: dict, step: int, log_prefix: str):
        if cls.verl_metric_map is not None:
            metrics = {**metrics, **cls._translate_to_verl(metrics)}
            # gcore logs ppo_step 0-based; verl ("we start from step 1") is
            # 1-based. Shift to verl's convention so curves overlay on the same x.
            step = step + 1
        metrics = reorder_dict_keys_by_prefix(metrics, "policy")
        if cls.tb_writer is not None:
            for key, val in metrics.items():
                cls.tb_writer.add_scalar(key, val, step)
        if cls.wandb_writer is not None:
            cls.wandb_writer.log(metrics, step=step, commit=True)

        metrics_text = " ".join(
            [f"{key} {value:.3e}" for key, value in metrics.items() if value is not None]
        )
        log(f"{log_prefix} {metrics_text}")

    @classmethod
    def finish(cls):
        if cls.tb_writer is not None:
            cls.tb_writer.close()
        if cls.wandb_writer is not None:
            cls.wandb_writer.finish()


def init_train_reporter_singleton(report_config, config):
    """Initialize TensorBoard and/or WandB writers based on configuration.

    Parameters
    ----------
    report_config : object
    config : object
        Logged to WandB.
    """
    log(f"init_train_reporter_singleton")
    if report_config.report_to in ["tensorboard", "both"]:
        TrainReporterSingleton._set_tensorboard_writer(report_config)
    if report_config.report_to in ["wandb", "both"]:
        TrainReporterSingleton._set_wandb_writer(report_config, config)
    if report_config.verl_metric_map_path is not None:
        import yaml
        map_path = report_config.verl_metric_map_path
        assert os.path.exists(map_path), f"verl_metric_map_path not found: {map_path}"
        with open(map_path) as f:
            TrainReporterSingleton.verl_metric_map = yaml.safe_load(f)
        log(f"loaded verl metric map from {map_path}")


def _ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)
        return path
    return None
