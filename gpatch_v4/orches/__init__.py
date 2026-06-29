import os

import ray
from ray.util.queue import Queue

from gpatch_v4.utils.logging_utils import (
    CAPTURE_INFER_ENGINE_LOG_ENV,
    DEBUG_LOG_TO_FILE_ENV,
    LOG_LEVEL_ENV,
    LOG_TO_DRIVER_ENV,
    TASK_LOG_DIR_ENV,
    setup_gpatch_logging,
)

PROPAGATE_ENV_KEYS = [
    "PYTHONPATH",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "PYTORCH_CUDA_ALLOC_CONF",
    "SGLANG_EMPTY_CACHE_INTERVAL",
    "SGLANG_RETURN_ORIGINAL_LOGPROB",
    TASK_LOG_DIR_ENV,
    LOG_LEVEL_ENV,
    DEBUG_LOG_TO_FILE_ENV,
    CAPTURE_INFER_ENGINE_LOG_ENV,
    LOG_TO_DRIVER_ENV,
    "GPATCH_LOG_STDIO",
    "GPATCH_ENGINE_LOG_FILE",
    "GPATCH_ENGINE_LOG_DIR",
    "GPATCH_ENGINE_ROLE",
    "VLLM_CUDART_SO_PATH",
]


def init(config=None):
    if config is not None:
        logger = setup_gpatch_logging(
            config,
            role="train_main",
            rank=0,
            force_new_log_dir=True,
            install_root=True,
            capture_stdio=True,
        )
        log_dir = os.environ[TASK_LOG_DIR_ENV]
        if hasattr(config, "report") and hasattr(config.report, "log_path"):
            config.report.log_path = log_dir

        message = f"[GCore logging] Redirect logs to path {log_dir}"
        print(message, flush=True)
        logger.info(message)

    # Driver aggregation relies on Ray forwarding actor stdout/stderr to the
    # driver, where train_main is the only writer of training.log.
    log_to_driver = True
    os.environ[LOG_TO_DRIVER_ENV] = "1"

    runtime_env_vars = {}
    for key in PROPAGATE_ENV_KEYS:
        val = os.environ.get(key)
        if val is not None:
            runtime_env_vars[key] = val

    # Ensure critical defaults even if not set on head node
    ray.init(
        log_to_driver=log_to_driver, namespace='train', runtime_env={"env_vars": runtime_env_vars}
    )


def shutdown():
    """Shut down the Ray runtime."""
    ray.shutdown()


def is_initialized():
    """Return whether the Ray runtime is initialized.

    Returns
    -------
    bool
    """
    return ray.is_initialized()


def get(*args, **kwargs):
    """Thin wrapper around ``ray.get``.

    Parameters
    ----------
    *args : object
    **kwargs : object

    Returns
    -------
    object
    """
    return ray.get(*args, **kwargs)


def get_actor(*args, **kwargs):
    """Thin wrapper around ``ray.get_actor``.

    Parameters
    ----------
    *args : object
    **kwargs : object

    Returns
    -------
    object
    """
    return ray.get_actor(*args, **kwargs)


def get_queue(maxsize: int):
    return Queue(maxsize=maxsize)


def get_node_ip() -> str:
    """Return the IP address of the current node.

    Returns
    -------
    str
    """
    return ray.util.get_node_ip_address()
