# Adapted from https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/trainer/ray/utils.py#L1
# Adapted from slime

import os
import socket

import netifaces
import ray
import torch

from gpatch_v4.core.device import get_visible_devices_env_var

# Refer to
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L102-L103
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py#L94-L95
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/hpu.py#L116-L117
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/neuron.py#L108-L109
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/tpu.py#L171-L172
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py#L97-L98
NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
    "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
    "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
    "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
]

_PROPAGATE_ENV_KEYS = [
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "PYTORCH_CUDA_ALLOC_CONF",
    "SGLANG_EMPTY_CACHE_INTERVAL",
    "SGLANG_RETURN_ORIGINAL_LOGPROB",
    "GPATCH_EXTRA_PROPAGATE_ENV",
    # deterministic mode
    "NCCL_DETERMINISTIC",
    "NCCL_ALGO",
    "FLASH_ATTENTION_DETERMINISTIC",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
    "CUBLAS_WORKSPACE_CONFIG",
    "GPATCH_DISABLE_FA3",
    "GPATCH_DYN_CP_CHECK_A2A",
]


def build_actor_env_vars(extra_env_vars=None):
    """Build env_vars dict for Ray actor runtime_env.

    Includes propagated env vars (PYTHONPATH, etc.) from the current
    process, any names additionally listed in ``GPATCH_EXTRA_PROPAGATE_ENV``
    (comma-separated, for task-local debug envs), NOSET_VISIBLE_DEVICES
    flags, and any caller-supplied extras.
    """
    env_vars = {}
    extra_keys = []
    extra_propagate = os.environ.get("GPATCH_EXTRA_PROPAGATE_ENV", "")
    if extra_propagate:
        extra_keys = [name.strip() for name in extra_propagate.split(",") if name.strip()]
    for key in (*_PROPAGATE_ENV_KEYS, *extra_keys):
        val = os.environ.get(key)
        if val is not None:
            env_vars[key] = val
    env_vars.update({name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST})
    if extra_env_vars:
        env_vars.update(extra_env_vars)
    print(f"build_actor_env_vars: {env_vars}", flush=True)
    return env_vars


def ray_noset_visible_devices(env_vars=os.environ):
    """Return *True* if any Ray no-set-visible-devices env var is active.

    Parameters
    ----------
    env_vars : Mapping, optional

    Returns
    -------
    bool
    """
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)


def get_physical_gpu_id():
    """Return the UUID string of the current CUDA device.

    Returns
    -------
    str
    """
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return str(props.uuid)


def is_port_available(port):
    """Return whether a TCP port is available for binding.

    Parameters
    ----------
    port : int

    Returns
    -------
    bool
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
            s.listen(1)
            return True
        except socket.error:
            return False
        except OverflowError:
            return False


def get_current_node_ip() -> str:
    """Return the IP address of the current node.

    Resolution order:

    1. ``__HOST_IP__`` environment variable
    2. IP of the ``bond1`` / ``bond0`` network interface
    3. IP of the ``eth0`` network interface
    4. ``127.0.0.1`` as last resort

    Returns
    -------
    str
    """
    host_ip = os.environ.get("__HOST_IP__")
    if host_ip:
        return host_ip

    for iface in ("bond1", "bond0", "eth0"):
        try:
            addrs = netifaces.ifaddresses(iface)
            ipv4 = addrs.get(netifaces.AF_INET)
            if ipv4:
                return ipv4[0]["addr"]
        except (ValueError, KeyError):
            continue

    return "127.0.0.1"


def find_free_port(start_port: int = 29600, consecutive: int = 1) -> int:
    """Find a free TCP port on this node, scanning from *start_port*.

    Parameters
    ----------
    start_port : int, optional
    consecutive : int, optional

    Returns
    -------
    int
        First port of a block of *consecutive* available ports.
    """
    port = start_port
    while not all(is_port_available(port + i) for i in range(consecutive)):
        port += 1
    return port


def get_local_gpu_id():
    """Determine the local GPU index visible to this process.

    Returns
    -------
    int
        Local GPU index, or ``-1`` when no GPU is assigned (e.g.
        CPU-only actors such as rule-only BT-RM).
    """
    gpu_ids = ray.get_gpu_ids()
    if not gpu_ids:
        return -1
    cvd = os.environ.get(get_visible_devices_env_var(), None)

    if cvd is None:
        return gpu_ids[0]
    else:
        return cvd.split(",").index(str(gpu_ids[0]))
