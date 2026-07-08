"""TensorBoard + Weights & Biases reporting facade.

参考 gcore-dev ``gpatch_v4/utils/report_utils.py`` 的 ``TrainReporterSingleton``：
- 单一对象同时持有 TB / W&B writer，按 ``report_to`` 决定启用哪些后端；
- 只在主进程（rank 0）真正写日志，其余 rank 为 no-op；
- W&B 支持自建 host 的 ``login`` 以及基于 run-id 文件的断点续传。

与 gcore 不同点：这里实现为**实例**而非 classmethod 单例，且提供与
``torch.utils.tensorboard.SummaryWriter`` 兼容的 ``add_scalar`` / ``flush`` /
``close`` 接口，方便在训练脚本里直接替换原有的 ``self.writer``。W&B 侧采用
「按 step 缓冲、flush 时一次性 commit」的策略，避免一个 step 内多次
``add_scalar`` 触发 W&B 的 step-commit 冲突。
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, Mapping, Optional

import torch.distributed as dist

from .config import ReportConfig

logger = logging.getLogger(__name__)


def _is_main_process() -> bool:
    """Rank 0 (or non-distributed single process)。"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


class Reporter:
    """统一的指标上报器，封装 TensorBoard 与 W&B。

    Args:
        config: :class:`ReportConfig`。
        run_config: 训练超参（dataclass / dict / argparse.Namespace），会作为
            ``wandb.init(config=...)`` 写入 W&B，便于在 UI 里查看本次运行配置。
        is_main_process: 是否在主进程。非主进程时所有方法为 no-op。
    """
    def __init__(
        self,
        config: ReportConfig,
        run_config: Any = None,
        *,
        is_main_process: Optional[bool] = None,
    ) -> None:
        self.config = config
        self.is_main_process = _is_main_process() if is_main_process is None else is_main_process
        self.tb_writer = None
        self.wandb = None
        # W&B per-step buffer：同一 step 内累积 scalar，flush 时一次性 log。
        self._wandb_buffer: Dict[str, float] = {}
        self._wandb_step: Optional[int] = None

        if not self.is_main_process:
            return
        if config.use_tensorboard:
            self._init_tensorboard()
        if config.use_wandb:
            self._init_wandb(run_config)

    # ── backend init ──────────────────────────────────────────────────
    def _init_tensorboard(self) -> None:
        if not self.config.tensorboard_dir:
            logger.warning(
                "[Reporter] report_to includes tensorboard but tensorboard_dir is unset; skip."
            )
            return
        from torch.utils.tensorboard import SummaryWriter

        os.makedirs(self.config.tensorboard_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(
            log_dir=self.config.tensorboard_dir,
            max_queue=self.config.tensorboard_queue_size,
        )

    def _init_wandb(self, run_config: Any) -> None:
        if not self.config.wandb_project:
            logger.warning("[Reporter] report_to includes wandb but wandb_project is empty; skip.")
            return
        import wandb

        save_dir = self.config.wandb_save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        run_id, resume = self._resolve_wandb_resume(wandb)

        # 自建实例需要显式 login(host)；官方 wandb.ai 走本机登录态。
        if self.config.wandb_key or self.config.wandb_host:
            wandb.login(key=self.config.wandb_key, host=self.config.wandb_host)

        wandb.init(
            dir=save_dir or None,
            name=self.config.wandb_exp_name,
            project=self.config.wandb_project,
            id=run_id,
            resume=resume,
            config=self._to_config_dict(run_config),
        )
        self.wandb = wandb
        logger.info(
            f"[Reporter] wandb initialized: project={self.config.wandb_project} "
            f"name={self.config.wandb_exp_name} id={run_id} resume={resume}"
        )

    # ── wandb resume (run-id 持久化) ──────────────────────────────────
    def _resolve_wandb_resume(self, wandb_module):
        """返回 ``(run_id, resume)``。

        若配置了 ``wandb_resume_dir``，在该目录维护 ``wandb_run_id.txt``：
        - 已存在则复用其中的 run id，并以 ``resume="allow"`` 续接同一 run；
        - 不存在则生成新 id 并落盘。
        否则使用显式 ``wandb_run_id``（可为 None）且不续传。
        """
        explicit_id = self.config.wandb_run_id
        resume_dir = self.config.wandb_resume_dir
        if not resume_dir:
            return explicit_id, None

        run_id_path = os.path.join(resume_dir, "wandb_run_id.txt")
        run_id = None
        if os.path.exists(run_id_path):
            with open(run_id_path, "r", encoding="utf-8") as f:
                run_id = f.read().strip() or None
            if run_id:
                logger.info(f"[Reporter] reuse wandb run id from {run_id_path}: {run_id}")

        if not run_id:
            run_id = explicit_id or self._generate_run_id(wandb_module)
            os.makedirs(resume_dir, exist_ok=True)
            tmp_path = f"{run_id_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(f"{run_id}\n")
            os.replace(tmp_path, run_id_path)
            logger.info(f"[Reporter] saved wandb run id to {run_id_path}: {run_id}")

        return run_id, "allow"

    @staticmethod
    def _generate_run_id(wandb_module) -> str:
        try:
            return wandb_module.util.generate_id()
        except Exception:
            return uuid.uuid4().hex[:8]

    @staticmethod
    def _to_config_dict(run_config: Any) -> Optional[dict]:
        if run_config is None:
            return None
        if isinstance(run_config, Mapping):
            return dict(run_config)
        if hasattr(run_config, "__dict__"):
            return dict(vars(run_config))
        return None

    # ── logging API ───────────────────────────────────────────────────
    def add_scalar(self, tag: str, scalar_value: Any, global_step: Optional[int] = None) -> None:
        """与 ``SummaryWriter.add_scalar`` 兼容；同时镜像到 W&B（按 step 缓冲）。"""
        if not self.is_main_process:
            return
        value = scalar_value.item() if hasattr(scalar_value, "item") else float(scalar_value)
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag, value, global_step=global_step)
        if self.wandb is not None:
            # step 切换时，先把上一个 step 的缓冲 commit 掉。
            if self._wandb_step is not None and global_step != self._wandb_step and self._wandb_buffer:
                self._flush_wandb()
            self._wandb_step = global_step
            self._wandb_buffer[tag] = value

    def log(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        """批量上报一组指标（一次性写入 TB + W&B）。"""
        if not self.is_main_process or not metrics:
            return
        if self.tb_writer is not None:
            for key, val in metrics.items():
                v = val.item() if hasattr(val, "item") else val
                self.tb_writer.add_scalar(key, v, global_step=step)
        if self.wandb is not None:
            payload = {k: (v.item() if hasattr(v, "item") else v) for k, v in metrics.items()}
            self.wandb.log(payload, step=step, commit=True)

    def _flush_wandb(self) -> None:
        if self.wandb is not None and self._wandb_buffer:
            self.wandb.log(self._wandb_buffer, step=self._wandb_step, commit=True)
            self._wandb_buffer = {}

    def flush(self) -> None:
        """与 ``SummaryWriter.flush`` 兼容；同时 commit W&B 缓冲。"""
        if not self.is_main_process:
            return
        self._flush_wandb()
        if self.tb_writer is not None:
            self.tb_writer.flush()

    def close(self) -> None:
        """关闭所有后端（``tb.close()`` + ``wandb.finish()``）。"""
        if not self.is_main_process:
            return
        self._flush_wandb()
        if self.tb_writer is not None:
            self.tb_writer.close()
            self.tb_writer = None
        if self.wandb is not None:
            self.wandb.finish()
            self.wandb = None

    # alias，对齐 gcore-dev 命名习惯
    finish = close


def build_reporter(
    config: ReportConfig,
    run_config: Any = None,
    *,
    is_main_process: Optional[bool] = None,
) -> Reporter:
    """构造 :class:`Reporter` 的工厂函数。"""
    return Reporter(config, run_config, is_main_process=is_main_process)
