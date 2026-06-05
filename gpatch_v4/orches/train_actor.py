import abc
import os
import random
import re
from datetime import timedelta

import ray
import torch

# isort: off
import gpatch_v4.core.device  # noqa: F401  # ensure device backend is initialized early
# isort: on

import torch.distributed as dist
from transformers import AutoConfig

import gpatch_v4.utils.common_utils as common_utils
from gpatch_v4.orches.base_actor import RayBaseActor
from gpatch_v4.orches.utils import get_local_gpu_id
from gpatch_v4.utils import monkey_patch_torch_dist
from gpatch_v4.utils.logging_utils import setup_gpatch_logging


class BaseActor(RayBaseActor):
    """Base distributed training actor.

    Sets up env vars for torch distributed and discovers master
    address / port for rank-0 actors.

    Parameters
    ----------
    world_size : int
    rank : int
    master_addr : str or None
        *None* for rank-0 (auto-discovered).
    master_port : int or None
        *None* for rank-0 (auto-discovered).
    """
    def __init__(self, world_size, rank, master_addr, master_port):
        self._world_size = world_size
        self._rank = rank
        if master_addr:
            self.master_addr, self.master_port = master_addr, master_port
        else:
            self.master_addr, self.master_port = self._get_current_node_ip_and_free_port(
                start_port=random.randint(10000, 29999)
            )

        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        self._local_gpu_id = get_local_gpu_id()
        os.environ["LOCAL_RANK"] = str(self._local_gpu_id)

    @property
    def has_gpu(self) -> bool:
        """Whether this actor was assigned a GPU by Ray."""
        return self._local_gpu_id >= 0

    def shutdown(self) -> None:
        """Graceful teardown hook, exposed as a remote method.

        No-op by default. Subclasses owning an sglang engine
        (``GrpoSamplerActor``, ``GrpoGenRmActor``, ``T2iGrpoGenRmActor``)
        override this to explicitly reap sglang's subprocess tree before
        the ray worker is SIGKILLed; otherwise scheduler / TP worker /
        detokenizer leak as PID-1 orphans.

        Callers MUST invoke ``actor.shutdown.remote()`` explicitly before
        ``ray.kill(actor)`` — we do NOT rely on Python GC / ``__del__``
        timing.
        """
        return

    def log_memory(self, tag: str, rank: int = 0):
        """Log detailed GPU memory usage on rank-0 of this actor."""
        from gpatch_v4.utils import logging_memory_usage_details
        logging_memory_usage_details(tag, rank=rank)

    def init(self, config, pg_backend='nccl'):
        """Initialize the actor with config and distributed backend.

        Parameters
        ----------
        config : object
        pg_backend : str, optional
        """
        self.config = config
        log_role = get_actor_log_role(self.__class__.__name__)
        self.logger = setup_gpatch_logging(
            config,
            role=log_role,
            rank=self._rank,
            console_rank=None,
            capture_stdio=False,
            install_root=True,
            log_to_driver=True,
        )
        common_utils.set_default_logger(self.logger)
        # infer_only_mode 仅推理不训练，兼容没有 policy 和 training 配置字段
        infer_only_mode = not hasattr(self.config,
                                      'policy') and not hasattr(self.config, 'training')
        if not infer_only_mode and self.config.training.offload_process_group:
            monkey_patch_torch_dist()

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(f"cuda:{local_rank}")

        dist.init_process_group(
            backend=pg_backend,
            timeout=timedelta(
                minutes=(
                    config.sampler.dist_config.torch_dist_timeout_minutes
                    if infer_only_mode else config.policy.dist_config.torch_dist_timeout_minutes
                )
            ),
        )

        try:
            import pynvml

            pynvml.nvmlInit()

            local_rank = int(os.environ["RANK"]) % (
                config.sampler.dist_config.num_gpus_per_node
                if infer_only_mode else config.policy.dist_config.num_gpus_per_node
            )  # infer_only_mode 仅推理不训练，兼容没有 policy 和 training 配置字段

            handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
            pynvml.nvmlDeviceSetCpuAffinity(handle)

            self.logger.info(f"set NUMA affinity for GPU {local_rank}")
            pynvml.nvmlShutdown()

        except ImportError:
            self.logger.warning("pynvml not available, skipping NUMA affinity setup")
        except Exception as e:
            self.logger.warning(f"failed to set NUMA affinity: {e}")

        from gpatch_v4.utils import debug_moe
        debug_moe.install(self.config)

    def set_logging_basic_config(self):
        """Configure Python logging with a standard format."""
        log_role = get_actor_log_role(self.__class__.__name__)
        self.logger = setup_gpatch_logging(
            self.config,
            role=log_role,
            rank=self._rank,
            console_rank=None,
            capture_stdio=False,
            install_root=True,
            log_to_driver=True,
        )
        common_utils.set_default_logger(self.logger)

    def load_hf_config(self):
        """Load a HuggingFace model config and attach it to the policy config."""
        if hasattr(self.config, 'policy'):
            hf_model_path = self.config.policy.hf_model_path
        else:
            # infer_only mode: no policy config, use sampler model_info instead
            hf_model_path = self.config.sampler.model_info[self.idx].hf_model_path
        hf_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
        # generate 可能用到模型相关的配置
        # attach hf_config on config for SamplerGenerateFunc to access
        if hasattr(self.config, 'policy'):
            assert not hasattr(self.config.policy, "hf_config")
            self.config.policy.hf_config = hf_config
        # also store on sampler model_info for infer_only access
        self.hf_config = hf_config


def get_actor_log_role(class_name: str) -> str:
    return _camel_to_snake(class_name)


def _camel_to_snake(name: str) -> str:
    first_pass = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first_pass).lower()
