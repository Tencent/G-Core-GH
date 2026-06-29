import gc
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import socket
import sys
import tarfile
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Optional

import psutil
import torch

from gpatch_v4.custom.registry import register_custom_module
from gpatch_v4.utils.logging_utils import (
    get_default_logger,
    log,
    log_debug,
    logging_rank0,
    logging_with_rank_and_datetime,
    set_default_logger,
)


def get_free_port():
    """Find and return a free TCP port on localhost.

    Returns
    -------
    int
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def find_process_using_port(ip, port):
    """Log the PID that currently occupies the given port.

    Parameters
    ----------
    ip : str
        Host IP (used only for logging).
    port : int
    """
    for conn in psutil.net_connections():
        if conn.laddr and conn.laddr.port == port:
            pid = conn.pid
            if pid:
                try:
                    p = psutil.Process(pid)
                    logging_with_rank_and_datetime(f"Port {port} is used by PID {pid} ({p.name()})")
                except Exception as e:
                    logging_with_rank_and_datetime(
                        f"Port {port} is used by PID {pid} (process info not available)"
                    )
                return
    logging_with_rank_and_datetime(f"{ip=} {port} is not used by any process.")


def clear_memory():
    """Synchronize CUDA, run garbage collection, and empty the CUDA cache."""
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def n_times_clear_memory(n: int = 1):
    for i in range(n):
        clear_memory()


def get_memory_usage():
    """Return current GPU and CPU memory usage in GB.

    Returns
    -------
    dict
        ``{'gpu_memory_mb': float, 'cpu_memory_mb': float}``.
    """
    gpu_memory = 0
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
    process = psutil.Process(os.getpid())
    cpu_memory = process.memory_info().rss / 1024**3
    return {"gpu_memory_mb": gpu_memory, "cpu_memory_mb": cpu_memory}


def gen_unique_id() -> str:
    global unique_id, unique_id_lock
    with unique_id_lock:
        id = unique_id
        unique_id += 1
    return f"rank_{torch.distributed.get_rank()}_unique_id_{id}"


def get_nvml_memory_info(gpu_id=0) -> dict:
    from gpatch_v4.core.device import is_cuda

    res = None
    if torch.cuda.is_available() and is_cuda():
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        res = f'total={meminfo.total/1024**3:.3f} GB, used={meminfo.used/1024**3:.3f} GB, free={meminfo.free/1024**3:.3f} GB'
        pynvml.nvmlShutdown()

    return res


def get_meminfo_str(simple_info: bool = False) -> str:
    mem_info = {}
    with open('/proc/meminfo', 'r') as f:
        for line in f.readlines():
            parts = line.split(':')
            assert len(parts) == 2
            key = parts[0].strip()
            value = float(parts[1].strip().replace('kB', '').strip())
            mem_info[key] = value

    # MemTotal - MemFree - (Buffers + Cached + SReclaimable - Shmem)
    used = mem_info["MemTotal"] - mem_info["MemFree"] - (
        mem_info["Buffers"] + mem_info["Cached"] + mem_info["SReclaimable"] - mem_info["Shmem"]
    )
    mem_info["used"] = used

    pick_keys = set(
        [
            "MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "Shmem", "KReclaimable",
            "Mapped", "AnonPages", "used"
        ]
    )
    meminfo_str = ""
    for key, value in mem_info.items():
        if not simple_info or key in pick_keys:
            # convert in fixed unit
            for unit in ['KB', 'MB', 'GB', 'TB']:
                if value < 1024.0:
                    value = f"{value:.2f}{unit}"
                    break
                value /= 1024.0
            if isinstance(value, float):
                value = f"{value:.2f}PB"
            meminfo_str += f"{key}:{value}  "

    return meminfo_str


# TODO rename sync_cuda_and_get_time
def sync_cuda_and_get_time():
    torch.cuda.synchronize()
    return time.time()


def logging_memory_usage(log_info: str, verbose: bool = False, rank: int = None, logger=None):
    """Log GPU memory allocation, reservation, and NVML info.

    Parameters
    ----------
    log_info : str
        Descriptive prefix.
    verbose : bool, optional
        Unused.
    rank : int, optional
        Log only on this rank.
    logger : logging.Logger, optional
    """
    torch.cuda.synchronize()
    message = (
        f"{log_info} allocated {torch.cuda.memory_allocated() / (1024**3):.3f} GB"
        f" reserved {torch.cuda.memory_reserved() / (1024**3):.3f} GB"
        f" in process {get_nvml_memory_info(torch.cuda.current_device())}"
    )
    logging_with_rank_and_datetime(message, rank=rank, logger=logger)


def current_process_pids() -> set[int]:
    """Return all namespace pids that can refer to the current process."""
    pids = {os.getpid()}
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith("NSpid:"):
                    pids.update(int(pid) for pid in line.split()[1:])
                    break
    except OSError:
        pass
    return pids


# Lazily-discovered NVML host pid for this process, keyed by ``cuda``
# device index. ``None`` means a probe was attempted and produced an
# inconclusive result; we cache that too to avoid retrying every call.
nvml_host_pid_cache: Dict[int, Optional[int]] = {}


def snapshot_nvml_process_memory(handle, pynvml_module) -> Dict[int, int]:
    """Return ``{host_pid: usedGpuMemory_bytes}`` for compute procs on ``handle``."""
    procs = pynvml_module.nvmlDeviceGetComputeRunningProcesses(handle)
    return {p.pid: (getattr(p, "usedGpuMemory", 0) or 0) for p in procs}


def probe_nvml_host_pid(device: int) -> Optional[int]:
    """Discover this process's NVML host pid by allocating a sized probe.

    NVML reports host pids; from an unprivileged pid namespace the host pid
    is unreachable via ``/proc``. We allocate a process-unique-sized probe on
    ``device`` and observe which NVML pid grew by that exact amount.

    Parameters
    ----------
    device : int
        CUDA device index (as seen by this process / ``CUDA_VISIBLE_DEVICES``).

    Returns
    -------
    Optional[int]
        Discovered NVML host pid; ``None`` if the probe was inconclusive
        (no match or multiple matches).
    """
    import pynvml

    granularity = 2 * 1024 * 1024  # NVML reports allocation rounded ~2 MiB
    base_bytes = 64 * 1024 * 1024  # large enough to bypass torch's caching
    # Make probe size unique per container-local pid so that simultaneous
    # peer probes on the same device produce distinguishable deltas.
    probe_bytes = base_bytes + (os.getpid() % 4096) * granularity
    probe_bytes = ((probe_bytes + granularity - 1) // granularity) * granularity

    cuda_device = torch.device("cuda", device)

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        # Force lazy CUDA context init so the baseline includes our context
        # overhead; otherwise the post-probe delta would be (probe + ctx).
        torch.cuda.synchronize()
        torch.empty(1, dtype=torch.uint8, device=cuda_device).fill_(0)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        before = snapshot_nvml_process_memory(handle, pynvml)
        probe = torch.empty(probe_bytes, dtype=torch.uint8, device=cuda_device)
        probe.fill_(0)
        torch.cuda.synchronize()
        after = snapshot_nvml_process_memory(handle, pynvml)
        del probe
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    finally:
        pynvml.nvmlShutdown()

    tolerance = granularity
    matches = [
        nvml_pid for nvml_pid, used_after in after.items()
        if abs((used_after - before.get(nvml_pid, 0)) - probe_bytes) <= tolerance
    ]
    return matches[0] if len(matches) == 1 else None


def get_nvml_host_pid(device: int) -> Optional[int]:
    """Cached wrapper around ``probe_nvml_host_pid``.

    The probe runs at most once per ``device`` per process. Set the env var
    ``GPATCH_DISABLE_HOST_PID_PROBE=1`` to skip the probe entirely.
    """
    if device in nvml_host_pid_cache:
        return nvml_host_pid_cache[device]
    if os.environ.get("GPATCH_DISABLE_HOST_PID_PROBE"):
        nvml_host_pid_cache[device] = None
        return None
    try:
        host_pid = probe_nvml_host_pid(device)
    except Exception:
        host_pid = None
    nvml_host_pid_cache[device] = host_pid
    return host_pid


def logging_memory_usage_details(log_info: str, rank: int = None, logger=None):
    """Log detailed GPU memory usage, including per-process NVML accounting."""
    from gpatch_v4.core.device import is_cuda

    if torch.distributed.is_initialized() and rank is not None:
        if torch.distributed.get_rank() != rank:
            return

    torch.cuda.synchronize()
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    free_cuda, total_cuda = torch.cuda.mem_get_info(device)

    if not torch.cuda.is_available() or not is_cuda():
        message = (
            f"[MEMORY] {log_info} allocated {allocated / (1024**3):.3f} GB"
            f" reserved {reserved / (1024**3):.3f} GB in process"
        )
        logging_with_rank_and_datetime(message, rank=rank, logger=logger)
        return

    import pynvml

    self_pids = current_process_pids()
    host_pid = get_nvml_host_pid(device)
    if host_pid is not None:
        self_pids.add(host_pid)
    proc_entries: list[tuple[int, int, bool]] = []  # (pid, used, is_self)
    self_used: Optional[int] = None
    other_used = 0
    all_used = 0
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        for proc in procs:
            used = getattr(proc, "usedGpuMemory", 0) or 0
            all_used += used
            is_self = proc.pid in self_pids
            if is_self:
                self_used = (self_used or 0) + used
            else:
                other_used += used
            proc_entries.append((proc.pid, used, is_self))
    finally:
        pynvml.nvmlShutdown()

    # Sort by used desc so largest consumers are visible first; mark self.
    proc_entries.sort(key=lambda e: e[1], reverse=True)
    proc_rows = [
        f"{entry_pid}{'*' if is_self else ''}={entry_used / 1024**3:.3f}GB"
        for entry_pid, entry_used, is_self in proc_entries
    ]

    if self_used is None:
        # Rare fallback: host-pid probe did not yield a matching NVML pid
        # this snapshot, so we cannot split ``all_used`` into self vs other.
        sentinel = "unknown(probe_failed)"
        self_nvml_str = sentinel
        self_non_torch_str = sentinel
        other_procs_str = sentinel
    else:
        self_nvml_str = f"{self_used / 1024**3:.3f} GB"
        self_non_torch_str = f"{(self_used - reserved) / 1024**3:.3f} GB"
        other_procs_str = f"{other_used / 1024**3:.3f} GB"

    # Field semantics (only the non-obvious parts):
    #   {allocated,reserved} : torch caching allocator view; excludes
    #       CUDA context and any non-torch GPU memory.
    #   self_nvml                  : this process's GPU memory per NVML
    #       (torch_reserved + self_non_torch).
    #   self_non_torch             : self_nvml - torch_reserved, i.e. CUDA
    #       context + NCCL / cuBLAS / cuDNN / flash-attn / TE workspaces +
    #       raw cudaMalloc (e.g. bucketed IPC staging) that torch can't see.
    #   other_procs                : sum of NVML compute procs that are not
    #       this process.
    #   all_compute_procs          = self_nvml + other_procs; usually a few
    #       hundred MB below ``device_used`` (driver overhead not attributed
    #       to any compute proc).
    #   self_pids                  : ns-local pid(s) + probe-discovered host
    #       pid (see ``_resolve_self_host_pid_on_device``).
    #   cuda_{total,free}          : torch.cuda.mem_get_info; may differ
    #       slightly from device_* due to CUDA_VISIBLE_DEVICES remapping.
    #   memory_per_procs_detail    : per-pid NVML used bytes, sorted desc by
    #       usage; self pid is tagged with a trailing ``*``.
    message = (
        f"[MEMORY] {log_info} "
        f"cuda_device={device} "
        f"allocated={allocated / 1024**3:.3f} GB "
        f"reserved={reserved / 1024**3:.3f} GB "
        f"self_nvml={self_nvml_str} "
        f"self_non_torch={self_non_torch_str} "
        f"other_procs={other_procs_str} "
        f"all_compute_procs={all_used / 1024**3:.3f} GB "
        f"self_pids={','.join(str(self_pid) for self_pid in sorted(self_pids))} "
        f"device_total={meminfo.total / 1024**3:.3f} GB "
        f"device_used={meminfo.used / 1024**3:.3f} GB "
        f"device_free={meminfo.free / 1024**3:.3f} GB "
        f"cuda_total={total_cuda / 1024**3:.3f} GB "
        f"cuda_free={free_cuda / 1024**3:.3f} GB | "
        f"memory_per_procs_detail=[{'; '.join(proc_rows)}]"
    )
    logging_with_rank_and_datetime(message, rank=rank, logger=logger)


def logging_meminfo_str(prefix_msg: str = "", logger=None):
    if not torch.distributed.is_initialized():
        if logger is None:
            logger = get_default_logger()
        logger.info(f"{prefix_msg} {get_meminfo_str(simple_info=True)}")
        return

    if 0 == torch.distributed.get_rank() % 8:
        logging_with_rank_and_datetime(
            f"{prefix_msg} {get_meminfo_str(simple_info=True)}", logger=logger
        )


@contextmanager
def profile_memory_and_time(stage: str, rank=None, logger=None):
    """Context manager that logs GPU/CPU memory delta and elapsed time.

    Parameters
    ----------
    stage : str
    rank : int, optional
        Log only on this rank.
    logger : logging.Logger, optional
    """
    torch.cuda.synchronize()
    start_time = time.time()
    start_memory = get_memory_usage()
    yield
    torch.cuda.synchronize()
    end_time = time.time()
    end_memory = get_memory_usage()

    logging_with_rank_and_datetime(
        f"\n{stage} "
        f" Time taken: {end_time - start_time:.2f} seconds \n"
        f" GPU memory before: {start_memory['gpu_memory_mb']:.2f} GB\n"
        f" GPU memory after: {end_memory['gpu_memory_mb']:.2f} GB\n"
        f" GPU memory delta: {end_memory['gpu_memory_mb'] - start_memory['gpu_memory_mb']:.2f} GB\n"
        f" CPU memory before: {start_memory['cpu_memory_mb']:.2f} GB\n"
        f" CPU memory after: {end_memory['cpu_memory_mb']:.2f} GB\n"
        f" CPU memory delta: {end_memory['cpu_memory_mb'] - start_memory['cpu_memory_mb']:.2f} GB",
        rank,
        logger=logger
    )


@contextmanager
def perf_time(stage: str, rank=None, logger=None):
    """Context manager that logs wall-clock time for a code block.

    Parameters
    ----------
    stage : str
    rank : int, optional
        Log only on this rank.
    logger : logging.Logger, optional
    """
    torch.cuda.synchronize()
    start_time = time.time()
    yield
    torch.cuda.synchronize()
    end_time = time.time()
    logging_with_rank_and_datetime(
        f"{stage} using time {end_time - start_time:.2f} seconds", rank, logger=logger
    )


def import_mod_from_path(py_path: str):
    """Dynamically import a Python module from a file path.

    Parameters
    ----------
    py_path : str
        Absolute path to a ``.py`` file.

    Returns
    -------
    module
    """
    assert py_path is not None and os.path.exists(py_path), f'invalid path {py_path}'
    import gpatch_v4.custom  # noqa: F401

    md5 = hashlib.md5()
    md5.update(py_path.encode('utf-8'))
    new_mod_name = md5.hexdigest()
    full_mod_name = f'gpatch_v4.custom.{new_mod_name}'
    if full_mod_name in sys.modules:
        new_mod = sys.modules[full_mod_name]
    else:
        spec = importlib.util.spec_from_file_location(full_mod_name, py_path)
        assert spec is not None, f"Failed to import module from path: {py_path}"
        new_mod = importlib.util.module_from_spec(spec)
        sys.modules[full_mod_name] = new_mod
        register_custom_module(new_mod_name, py_path)
        spec.loader.exec_module(new_mod)
    return new_mod


def import_fn_from_path(py_path: str, fn_name: str):
    """Import a specific function from a Python file.

    Parameters
    ----------
    py_path : str
        Absolute path to a ``.py`` file.
    fn_name : str

    Returns
    -------
    callable
    """
    mod = import_mod_from_path(py_path)
    assert hasattr(mod, fn_name), f"{mod} no such fn {py_path=} {fn_name=}"
    fn = getattr(mod, fn_name)
    return fn


def reorder_dict_keys_by_prefix(dict_data: Dict, prefix=None):
    """Reorder dict keys so keys starting with ``prefix`` come first.

    Parameters
    ----------
    dict_data : dict
    prefix : str, optional

    Returns
    -------
    dict
    """
    if prefix is None:
        return dict_data

    prefix_keys = [key for key in dict_data.keys() if key.startswith(prefix)]
    other_keys = [key for key in dict_data.keys() if not key.startswith(prefix)]

    ordered_keys = prefix_keys + other_keys
    return {key: dict_data[key] for key in ordered_keys}


HF_METADATA_FILE_SUFFIXES = (".json", ".py", ".txt", ".jinja")
HF_METADATA_EXCLUDED_FILES = {"model.safetensors.index.json"}
HF_METADATA_CACHE_ROOT = "/tmp/gcore_train_file"


def is_rank0():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def _is_hf_metadata_file(filename: str, include_config_json: bool) -> bool:
    if filename in HF_METADATA_EXCLUDED_FILES:
        return False
    if filename == "config.json":
        return include_config_json
    return filename.endswith(HF_METADATA_FILE_SUFFIXES)


def copy_hf_metadata_files(src_path, dst_path, include_config_json: bool):
    os.makedirs(dst_path, exist_ok=True)
    for item in os.listdir(src_path):
        if not _is_hf_metadata_file(item, include_config_json):
            continue
        src_file = os.path.join(src_path, item)
        if not os.path.isfile(src_file):
            continue
        dst_file = os.path.join(dst_path, item)
        logging_rank0(f"Copy {src_file} to {dst_file}.")
        shutil.copy(src_file, dst_file)


def get_hf_metadata_cache_dir(hf_model_path, save_path, cache_tag: str = "policy") -> str:
    """Return the local cache dir for HuggingFace metadata files."""
    source_key = os.path.abspath(os.path.expanduser(str(hf_model_path)))
    save_key = os.path.abspath(os.path.expanduser(str(save_path)))
    digest = hashlib.sha256(f"{source_key}|{save_key}".encode("utf-8")).hexdigest()[:16]
    return os.path.join(HF_METADATA_CACHE_ROOT, digest, cache_tag)


def cache_hf_metadata_files(hf_model_path, save_path, cache_tag: str = "policy") -> str:
    """Snapshot HuggingFace metadata files into local tmp storage.

    The cache is created at startup so later checkpoint export does not depend
    on the original HF directory still existing.
    """
    cache_dir = get_hf_metadata_cache_dir(hf_model_path, save_path, cache_tag)
    if not is_rank0():
        return cache_dir

    src_path = os.path.abspath(os.path.expanduser(str(hf_model_path)))
    assert os.path.isdir(src_path), f"Cannot cache HF metadata because {src_path} does not exist"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    copy_hf_metadata_files(src_path, cache_dir, include_config_json=True)
    return cache_dir


def assert_hf_metadata_cache_exists(hf_model_path, save_path, cache_tag: str = "policy") -> str:
    """Assert the local HuggingFace metadata cache exists."""
    cache_dir = get_hf_metadata_cache_dir(hf_model_path, save_path, cache_tag)
    if not is_rank0():
        return cache_dir

    assert os.path.isdir(cache_dir), (
        f"HF metadata cache {cache_dir} does not exist. "
        "cache_hf_metadata_files must run before save_hf."
    )
    return cache_dir


def copy_cached_hf_metadata_files(hf_model_path, save_path, dst_path, cache_tag: str = "policy"):
    """Copy cached HuggingFace metadata files into an exported checkpoint."""
    if not is_rank0():
        return

    cache_dir = assert_hf_metadata_cache_exists(hf_model_path, save_path, cache_tag)
    copy_hf_metadata_files(cache_dir, dst_path, include_config_json=True)


def copy_extra_file(src_path, dst_path):
    """Copy non-model files (JSON, Python, txt, jinja) between directories.

    Parameters
    ----------
    src_path : str
    dst_path : str
    """
    if not is_rank0():
        return
    copy_hf_metadata_files(src_path, dst_path, include_config_json=False)


def save_args_json(config, export_folder):
    """Save the config dataclass as a ``.gcore`` JSON file.

    Parameters
    ----------
    config : dataclass
    export_folder : str
    """
    if torch.distributed.get_rank() == 0:
        with open(os.path.join(export_folder, ".gcore"), "w") as f:
            json.dump(asdict(config), f)


def _compress_task(compressed_dir_name, compressed_dir, tar_path):
    with tarfile.open(tar_path, 'w:gz') as tar:
        tar.add(compressed_dir, arcname=compressed_dir_name, recursive=True)
    if not os.path.exists(tar_path) or os.path.getsize(tar_path) == 0:
        raise RuntimeError(f"Pack failed: missing {tar_path} or tar file empty")
    if os.path.exists(tar_path):
        shutil.rmtree(compressed_dir)


def compress_ppo_save_train_data(compact_thread, path):
    """Compress training data in a background thread.

    Parameters
    ----------
    compact_thread : threading.Thread or None
        Previous compression thread to join.
    path : str
        Directory containing training data.

    Returns
    -------
    threading.Thread or None
    """
    try:
        if compact_thread is not None:
            compact_thread.join()
        tmp_dir = os.path.join(path, 'tmp')
        if os.path.exists(tmp_dir) and len(os.listdir(tmp_dir)) >= 100:
            compressed_dir_name = f'train_info_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            compressed_dir = os.path.join(path, compressed_dir_name)  # 被压缩的目录
            os.rename(tmp_dir, compressed_dir)  # 打包完成后，会删除原文件。重命名文件夹防止误删新生成的ppo-train-info
            tar_path = os.path.join(path, f'{compressed_dir_name}.tar')  # 压缩后的.tar文件
            compact_thread = threading.Thread(
                target=_compress_task,
                args=(compressed_dir_name, compressed_dir, tar_path),
                daemon=False
            )
            compact_thread.start()
            return compact_thread
    except Exception as e:
        logging_with_rank_and_datetime(f" [DUMP METRICS] Train Infos Compress Failed : {e} !!!")
    return compact_thread


def can_import_class(class_path: str) -> bool:
    """Check whether a dotted class path is importable.

    Parameters
    ----------
    class_path : str
        Fully-qualified class path (e.g. ``'foo.bar.Baz'``).

    Returns
    -------
    bool
    """
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return hasattr(module, class_name)
    except (ImportError, ValueError, AttributeError):
        return False


def safe_import_class(class_path: str) -> Optional[Any]:
    """Import a class by dotted path, returning ``None`` on failure.

    Parameters
    ----------
    class_path : str

    Returns
    -------
    type or None
    """
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls
    except (ImportError, ValueError, AttributeError):
        logging_with_rank_and_datetime(
            f"Failed to import class '{class_path}'. Original traceback:\n{traceback.format_exc()}"
        )
        return None


def format_config(config, indent=2):
    """Format a dataclass config as a pretty-printed JSON string.

    Parameters
    ----------
    config : dataclass or dict
    indent : int, optional

    Returns
    -------
    str
    """
    import dataclasses
    if dataclasses.is_dataclass(config):
        d = asdict(config)
    elif isinstance(config, dict):
        d = config
    else:
        return str(config)

    def _default(obj):
        """Handle non-serializable objects."""
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        return str(obj)

    return json.dumps(d, indent=indent, default=_default, ensure_ascii=False)


# copy from: sglang/python/sglang/srt/utils/common.py
def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
    """Kill the process and all its child processes."""
    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    if include_parent:
        try:
            if parent_pid == os.getpid():
                itself.kill()
                sys.exit(0)

            itself.kill()

            # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
            # so we send an additional signal to kill them.
            itself.send_signal(signal.SIGQUIT)
        except psutil.NoSuchProcess:
            pass


class Envelope:
    """
    从 slime 参考来的写法，防止 Ray 自动解引用 ObjectRef。

    ray.put(rollout_data) 返回一个 ObjectRef。如果你把裸的 ObjectRef 放在列表里，然后通过 Ray remote 调用传给另一个 actor/task，
    Ray 会自动对 ObjectRef 执行 ray.get()，把数据提前拉到调用方内存里——这不是我们想要的行为，因为：

    数据应该留在 object store 里，只在真正需要它的 actor（按 DP rank 取自己那份）才 ray.get()
    1. 如果不包一层，所有 DP rank 的数据都会在传参时被序列化到发送方，白白占用内存和带宽
    2. Box 把 ObjectRef 藏在一个普通 Python 对象的属性里，Ray 序列化 Box 时不会递归解引用内部的 ObjectRef。接收方通过
    rollout_data_ref[dp_rank].inner 拿到原始 ObjectRef，再显式 ray.get() 按需取数据。
    """
    def __init__(self, inner_data):
        self._inner_data = inner_data

    @property
    def inner_data(self):
        return self._inner_data
