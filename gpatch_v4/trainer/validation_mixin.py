"""Shared in-training validation hook for Bagel / WGOv3 trainers."""

import os
import traceback

import torch.distributed as dist

from gpatch_v4.utils.report_utils import TrainReporterSingleton


class InTrainingValidationMixin:
    """Provides shared scheduling, execution, and reporting for validation."""

    VALIDATION_WANDB_IMAGES = ()

    def build_validation_runner(self):
        """Return a validation runner instance, or None if unsupported."""
        return None

    def _should_validate(self, curr_step: int, force: bool = False) -> bool:
        training_args = self.training_args
        if not getattr(training_args, "validation_data_dir", ""):
            return False
        if force:
            return True
        val_every = getattr(training_args, "validation_every", 0)
        return val_every > 0 and curr_step > 0 and curr_step % val_every == 0

    def _get_validation_runner(self):
        if self._val_runner is None:
            self._val_runner = self.build_validation_runner()
        return self._val_runner

    def _log_validation_metrics(self, curr_step: int, metrics: dict, save_root: str):
        if dist.get_rank() != 0 or not metrics:
            return

        self.logger.info(f"[Validation] step={curr_step} metrics:")
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")

        wandb_writer = TrainReporterSingleton.get_wandb_writer()
        if wandb_writer is None:
            TrainReporterSingleton.log_and_report(metrics, curr_step, log_prefix="val")
            return

        wandb_payload = dict(metrics)
        for model_tag in ("train", "ema"):
            model_dir = os.path.join(save_root, model_tag)
            if not os.path.isdir(model_dir):
                continue
            for img_name in getattr(self, "VALIDATION_WANDB_IMAGES", ()):
                img_path = os.path.join(model_dir, f"{img_name}.png")
                if os.path.isfile(img_path):
                    wandb_payload[f"val/{model_tag}/{img_name}"] = wandb_writer.Image(img_path)

        wandb_writer.log(wandb_payload, step=curr_step)

    def maybe_validate(self, curr_step: int, force: bool = False):
        if not self._should_validate(curr_step, force=force):
            return False

        runner = self._get_validation_runner()
        if runner is None:
            return False
        self.cpu_barrier()
        self.logger.info(f"[Validation] Starting at step {curr_step}...")
        try:
            metrics, save_root = runner.run(curr_step)
        except Exception as exc:
            self.logger.info(f"[Validation] Failed: {exc}")
            traceback.print_exc()
            return False
        finally:
            self.cpu_barrier()

        self._log_validation_metrics(curr_step, metrics, save_root)
        return True
