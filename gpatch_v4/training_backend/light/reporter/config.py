"""Configuration for experiment-metric reporting (TensorBoard / Weights & Biases).

参考 gcore-dev ``gpatch_v4/configs/report_config.py`` 的 ``ReportConfig`` 设计，
裁剪为本仓库训练脚本需要的字段。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

_VALID_REPORT_TO = {"none", "tensorboard", "wandb", "both"}


@dataclass
class ReportConfig:
    """Reporting backend configuration.

    Attributes:
        report_to: 上报后端，取值 ``none`` | ``tensorboard`` | ``wandb`` | ``both``。
        tensorboard_dir: TensorBoard 日志目录；为 ``None`` 时禁用 TensorBoard。
        tensorboard_queue_size: ``SummaryWriter`` 的 ``max_queue``。
        wandb_key: W&B API key（自建实例为 ``local-...``）；缺省走 wandb 本机登录态。
        wandb_host: 自建 W&B 服务地址；为 ``None`` 时使用官方 wandb.ai。
        wandb_project: W&B project 名。
        wandb_exp_name: W&B run 显示名（``wandb.init(name=...)``）。
        wandb_run_id: 显式指定 run id；与 ``wandb_resume_dir`` 配合做断点续传。
        wandb_save_dir: W&B 本地缓存目录（``wandb.init(dir=...)``）。
        wandb_resume_dir: 持久化 run id 的目录（通常为 checkpoint 输出目录）。
            非空时会在该目录写 ``wandb_run_id.txt``，resume 时复用同一 run。
    """

    report_to: str = "tensorboard"

    # TensorBoard
    tensorboard_dir: Optional[str] = None
    tensorboard_queue_size: int = 10

    # Weights & Biases
    wandb_key: Optional[str] = None
    wandb_host: Optional[str] = None
    wandb_project: str = "wegen-video"
    wandb_exp_name: Optional[str] = None
    wandb_run_id: Optional[str] = None
    wandb_save_dir: str = "wandb_local"
    wandb_resume_dir: Optional[str] = None

    def __post_init__(self) -> None:
        if self.report_to not in _VALID_REPORT_TO:
            raise ValueError(f"report_to must be one of {_VALID_REPORT_TO}, got '{self.report_to}'")

    @property
    def use_tensorboard(self) -> bool:
        return self.report_to in ("tensorboard", "both")

    @property
    def use_wandb(self) -> bool:
        return self.report_to in ("wandb", "both")
