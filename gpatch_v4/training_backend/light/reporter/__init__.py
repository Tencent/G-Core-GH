"""Experiment-metric reporting (TensorBoard + Weights & Biases).

    典型用法::

    from light.reporter import ReportConfig, build_reporter

    reporter = build_reporter(
        ReportConfig(report_to="both", tensorboard_dir=log_dir,
                     wandb_project="wegen-video", wandb_exp_name=run_name,
                     wandb_resume_dir=output_dir),
        run_config=args,
    )
    reporter.add_scalar("loss/train", loss, global_step=step)
    reporter.flush()
    reporter.close()
"""

from .config import ReportConfig
from .reporter import Reporter, build_reporter

__all__ = ["ReportConfig", "Reporter", "build_reporter"]
