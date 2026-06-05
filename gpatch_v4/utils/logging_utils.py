import logging
import os
import re
import socket
import sys
from collections.abc import Collection
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

TASK_LOG_DIR_ENV = "GPATCH_TASK_LOG_DIR"
LOG_STDIO_ENV = "GPATCH_LOG_STDIO"
LOG_LEVEL_ENV = "GPATCH_LOG_LEVEL"
DEBUG_LOG_TO_FILE_ENV = "GPATCH_DEBUG_LOG_TO_FILE"
CAPTURE_INFER_ENGINE_LOG_ENV = "GPATCH_CAPTURE_INFER_ENGINE_LOG"
LOG_TO_DRIVER_ENV = "GPATCH_LOG_TO_DRIVER"
DEBUG_LOG_DIRNAME = "debug_log"

DEFAULT_LOGGER_NAME = "gpatch_v4"
stdio_redirected = False
debug_logging_enabled = False
debug_file_logging_enabled = False

STDERR_ERROR_PATTERNS = re.compile(
    r"(Traceback|Exception|Error|CUDA error|NCCL|SIGSEGV|SIGBUS|"
    r"RuntimeError|ValueError|AssertionError|ray::)",
    re.IGNORECASE,
)
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
GPATCH_RECORD_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - "
    r"(DEBUG|INFO|WARNING|ERROR|CRITICAL) - role="
)
RAY_PREFIX_RE = re.compile(r"^(?P<prefix>\([^)]*(?:pid=|ip=)[^)]*\)\s*)(?P<message>.*)$")


def create_default_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


default_logger = create_default_logger()


def get_default_logger() -> logging.Logger:
    return default_logger


def set_default_logger(logger: logging.Logger) -> None:
    global default_logger
    default_logger = logger


def set_debug_logging_enabled(enabled: bool) -> None:
    global debug_logging_enabled
    debug_logging_enabled = enabled


def set_debug_file_logging_enabled(enabled: bool) -> None:
    global debug_file_logging_enabled
    debug_file_logging_enabled = enabled


def logging_rank0(message, logger: Optional[logging.Logger] = None) -> None:
    logger = logger or default_logger
    dist = get_torch_dist()
    if dist is not None and dist.is_initialized():
        if dist.get_rank() == 0:
            my_rank = dist.get_rank()
            logger.info(f"[RANK {my_rank:<4}] {message}")
    else:
        logger.info(f"{message}")


def logging_with_rank_and_datetime(
    message,
    rank: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    logger = logger or default_logger
    dist = get_torch_dist()
    if dist is not None and dist.is_initialized():
        my_rank = dist.get_rank()
        if rank is None or my_rank == rank:
            logger.info(f"[RANK {my_rank:<4}] {message}")
    else:
        logger.info(message)


log = logging_with_rank_and_datetime


def log_debug(
    message,
    rank: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    if not (debug_logging_enabled or debug_file_logging_enabled):
        return
    logger = logger or default_logger
    dist = get_torch_dist()
    if dist is not None and dist.is_initialized():
        my_rank = dist.get_rank()
        if rank is None or my_rank == rank:
            logger.debug(f"[RANK {my_rank:<4}] {message}")
    else:
        logger.debug(message)


def log_info(
    message,
    rank: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    logger = logger or default_logger
    dist = get_torch_dist()
    if dist is not None and dist.is_initialized():
        my_rank = dist.get_rank()
        if rank is None or my_rank == rank:
            logger.info(f"[RANK {my_rank:<4}] {message}")
    else:
        logger.info(message)


def get_torch_dist():
    try:
        import torch.distributed as dist
    except Exception:
        return None
    return dist


class GpatchContextFilter(logging.Filter):
    def __init__(self, role: str, rank: Optional[int]):
        super().__init__()
        self.role = normalize_role_alias(role)
        self.rank = rank_to_str(rank)
        self.pid = os.getpid()
        self.node = get_node_ip()

    def filter(self, record):
        record.gpatch_role = self.role
        record.gpatch_rank = self.rank
        record.gpatch_pid = self.pid
        record.gpatch_node = self.node
        return True


class TrainingLevelFilter(logging.Filter):
    def __init__(self, include_debug: bool):
        super().__init__()
        self.include_debug = include_debug

    def filter(self, record):
        if self.include_debug:
            return True  # DEBUG and above
        return record.levelno >= logging.INFO


class DebugLevelFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.DEBUG


def atomic_append_to_fd(fd: int, blob: bytes) -> None:
    """Append one complete record.

    ``training.log`` is written only by the driver process; actor processes
    write only optional per-process debug shards, so no cross-node file lock
    is needed on the hot logging path.
    """
    write_all(fd, blob)


def write_all(fd: int, blob: bytes) -> None:
    view = memoryview(blob)
    off = 0
    total = len(view)
    while off < total:
        try:
            written = os.write(fd, view[off:])
        except InterruptedError:
            continue
        if written <= 0:
            break
        off += written


class AtomicRecordHandler(logging.Handler):
    def __init__(self, path: str):
        super().__init__()
        self.log_path = path
        Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)

    @property
    def baseFilename(self) -> str:
        return self.log_path

    def fileno(self) -> int:
        return self.fd

    def emit(self, record):
        try:
            blob = (self.format(record) + "\n").encode("utf-8", errors="replace")
            atomic_append_to_fd(self.fd, blob)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        return

    def close(self) -> None:
        # Do NOT close the fd. Third-party libraries (e.g. vLLM) call
        # logging.config.dictConfig() which triggers _clearExistingHandlers()
        # and closes all handlers. We intentionally leak the fd so that
        # re-installed handlers and TeeStream can continue writing.
        super().close()


class TeeStream:
    """Intercept driver stdout/stderr and write selected records to training.log."""
    def __init__(
        self,
        original,
        training_log_fd: int,
        stream_name: str,
        role: str,
        rank: Optional[int],
    ):
        self.original = original
        self.training_log_fd = training_log_fd
        self.stream_name = stream_name
        self.role = normalize_role_alias(role)
        self.rank = rank_to_str(rank)
        self.pid = os.getpid()
        self.node = get_node_ip()
        self._in_error_block = False
        self._line_buffer = ""

    def write(self, data):
        if not data:
            return 0
        # Always forward to original stream (console)
        if self.original is not None:
            self.original.write(data)
            self.original.flush()
        self._capture_lines(data)
        return len(data)

    def flush(self):
        self._flush_line_buffer()
        if self.original is not None:
            self.original.flush()

    def fileno(self):
        """Return the underlying fd for APIs that require it (e.g. faulthandler)."""
        if self.original is not None and hasattr(self.original, "fileno"):
            try:
                return self.original.fileno()
            except Exception:
                pass
        return self.training_log_fd

    def isatty(self):
        return False

    @property
    def encoding(self):
        return getattr(self.original, "encoding", "utf-8")

    def _format_error_record(self, data: str) -> bytes:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        header = (
            f"{timestamp} - ERROR - role={self.role} - rank={self.rank} - "
            f"pid={self.pid} - node={self.node} - stream={self.stream_name}"
        )
        lines = data.splitlines() or [data]
        parts = [f"{header} - {lines[0]}\n"]
        for line in lines[1:]:
            parts.append(f"  {line}\n")
        return "".join(parts).encode("utf-8", errors="replace")

    def _write_to_training_log(self, data: str) -> None:
        atomic_append_to_fd(self.training_log_fd, self._format_error_record(data))

    def _capture_lines(self, data: str) -> None:
        self._line_buffer += data
        lines = self._line_buffer.split("\n")
        self._line_buffer = lines[-1]
        for line in lines[:-1]:
            self._capture_line(line.rstrip("\r"))

    def _flush_line_buffer(self) -> None:
        if not self._line_buffer:
            return
        line = self._line_buffer
        self._line_buffer = ""
        self._capture_line(line.rstrip("\r"))

    def _capture_line(self, line: str) -> None:
        stripped_line = ANSI_ESCAPE_RE.sub("", line)
        ray_match = RAY_PREFIX_RE.match(stripped_line)
        if ray_match:
            self._write_ray_forwarded_line(ray_match)
            return

        # Driver-local stderr errors are still useful even though bare driver
        # stdout is not mirrored to avoid duplicating train_main logger output.
        if self.stream_name != "stderr":
            return
        if STDERR_ERROR_PATTERNS.search(line):
            self._in_error_block = True
            self._write_to_training_log(line)
        elif self._in_error_block:
            if line.strip() == "":
                self._in_error_block = False
            self._write_to_training_log(line)

    def _write_ray_forwarded_line(self, ray_match: re.Match) -> None:
        prefix = ray_match.group("prefix")
        message = ray_match.group("message")
        if GPATCH_RECORD_RE.match(message):
            record = message
        else:
            level = "ERROR" if self.stream_name == "stderr" else "INFO"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
            record = (
                f"{timestamp} - {level} - role=ray_actor - rank=none - "
                f"pid={self.pid} - node={self.node} - stream={self.stream_name} - "
                f"{prefix}{message}"
            )
        atomic_append_to_fd(
            self.training_log_fd,
            (record + "\n").encode("utf-8", errors="replace"),
        )


def derive_task_log_dir(config=None, force_new: bool = False) -> str:
    """Derive or create a task-wise log directory with a timestamp suffix."""
    env_dir = os.environ.get(TASK_LOG_DIR_ENV)
    if env_dir and not force_new:
        return env_dir

    run_name = first_non_empty(
        nested_get(config, "report", "wandb_exp_name"),
        nested_get(config, "training", "run_name"),
        nested_get(config, "experiment_name"),
        nested_get(config, "name"),
    )
    run_name = sanitize_path_component(str(run_name)) if run_name else "task"
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_log_root = nested_get(config, "report", "log_dir")
    if report_log_root:
        log_dir = os.path.join(str(report_log_root), f"{run_name}_{suffix}")
    else:
        checkpoint_root = first_non_empty(
            nested_get(config, "checkpoint", "save_ckpt_path"),
            nested_get(config, "training", "checkpoint_dir"),
            nested_get(config, "checkpoint_dir"),
        )
        if checkpoint_root:
            log_dir = os.path.join(str(checkpoint_root), "logs", f"{run_name}_{suffix}")
        else:
            log_dir = os.path.join("logs", f"{run_name}_{suffix}")

    log_dir = os.path.abspath(log_dir)
    os.environ[TASK_LOG_DIR_ENV] = log_dir
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    return log_dir


def setup_gpatch_logging(
    config=None,
    role: str = "train_main",
    rank: Optional[int] = None,
    log_dir: Optional[str] = None,
    console_rank: Optional[Union[int, Collection[int]]] = 0,
    capture_stdio: bool = False,
    force_new_log_dir: bool = False,
    install_root: bool = False,
    log_to_driver: bool = False,
) -> logging.Logger:
    """Configure task-wise, level-based logging for a process."""
    if rank is None:
        rank = get_current_rank()
    if log_dir is None:
        log_dir = derive_task_log_dir(config, force_new=force_new_log_dir)
    os.environ[TASK_LOG_DIR_ENV] = log_dir

    log_level = resolve_log_level(config)
    debug_to_file = resolve_debug_log_to_file(config)
    capture_infer_engine_log = resolve_capture_infer_engine_log(config)
    include_debug_in_training = log_level <= logging.DEBUG
    set_debug_logging_enabled(include_debug_in_training)
    set_debug_file_logging_enabled(debug_to_file)
    os.environ[LOG_LEVEL_ENV] = logging.getLevelName(log_level).lower()
    os.environ[DEBUG_LOG_TO_FILE_ENV] = "1" if debug_to_file else "0"
    os.environ[CAPTURE_INFER_ENGINE_LOG_ENV] = "1" if capture_infer_engine_log else "0"
    os.environ[LOG_TO_DRIVER_ENV] = "1" if log_to_driver else "0"

    logger = logging.getLogger(f"{DEFAULT_LOGGER_NAME}.{sanitize_logger_name(role)}")
    logger.setLevel(logging.DEBUG if include_debug_in_training or debug_to_file else logging.INFO)
    logger.propagate = False
    console_level = logging.DEBUG if include_debug_in_training else logging.INFO
    if log_to_driver:
        clear_level_file_handlers(logger)
        ensure_console_handler(logger, role, rank, level=console_level, use_current_stderr=True)
        if debug_to_file:
            ensure_debug_file_handler(logger, log_dir, role, rank)
    else:
        install_level_handlers(
            logger, log_dir, role, rank, include_debug_in_training, debug_to_file
        )

    if install_root:
        root_logger = logging.getLogger()
        root_logger.setLevel(
            logging.DEBUG if include_debug_in_training or debug_to_file else logging.INFO
        )
        if log_to_driver:
            clear_level_file_handlers(root_logger)
            clear_plain_console_handlers(root_logger)
            ensure_console_handler(
                root_logger, role, rank, level=console_level, use_current_stderr=True
            )
            if debug_to_file:
                ensure_debug_file_handler(root_logger, log_dir, role, rank)
        else:
            install_level_handlers(
                root_logger, log_dir, role, rank, include_debug_in_training, debug_to_file
            )

    if not log_to_driver and should_log_to_console(rank, console_rank):
        ensure_console_handler(logger, role, rank, level=console_level)
    if capture_stdio and not log_to_driver:
        redirect_stdio_to_level_logs(log_dir, role=role, rank=rank)

    install_compat_logger(logger)
    install_excepthook(logger)
    logger.info("logging initialized at %s", os.path.abspath(log_dir))
    return logger


def should_log_to_console(
    rank: Optional[int],
    console_rank: Optional[Union[int, Collection[int]]],
) -> bool:
    if console_rank is None or rank is None:
        return True
    if isinstance(console_rank, int):
        return rank == console_rank
    return rank in console_rank


def configure_third_party_logging(
    role: str,
    backend: str,
    rank: Optional[int] = None,
    engine_idx: Optional[int] = None,
    rm_idx: Optional[int] = None,
    log_dir: Optional[str] = None,
) -> Optional[str]:
    """Route third-party Python logging into task-wise level log files.

    Re-installs handlers on ``gpatch_v4`` loggers because third-party
    libraries (e.g. vLLM) call ``logging.config.dictConfig()`` which removes
    all existing handlers via ``_clearExistingHandlers()``.
    """
    if rank is None:
        rank = get_current_rank()
    if log_dir is None:
        log_dir = os.environ.get(TASK_LOG_DIR_ENV) or derive_task_log_dir()
    log_level = resolve_log_level()
    debug_to_file = resolve_debug_log_to_file()
    include_debug_in_training = log_level <= logging.DEBUG
    log_to_driver = resolve_log_to_driver()
    console_level = logging.DEBUG if include_debug_in_training else logging.INFO
    root_logger = logging.getLogger()
    root_logger.setLevel(
        logging.DEBUG if include_debug_in_training or debug_to_file else logging.INFO
    )
    handler_role = f"{backend}_{role}"
    if log_to_driver:
        clear_level_file_handlers(root_logger)
        clear_plain_console_handlers(root_logger)
        ensure_console_handler(
            root_logger, handler_role, rank, level=console_level, use_current_stderr=True
        )
        if debug_to_file:
            ensure_debug_file_handler(root_logger, log_dir, handler_role, rank)
    else:
        install_level_handlers(
            root_logger, log_dir, handler_role, rank, include_debug_in_training, debug_to_file
        )
    # Re-install handlers on gpatch_v4 child loggers that may have been
    # cleared by third-party dictConfig calls.
    for name, logger_ref in logging.Logger.manager.loggerDict.items():
        if not isinstance(logger_ref, logging.Logger):
            continue
        if not name.startswith(DEFAULT_LOGGER_NAME):
            continue
        logger_ref.setLevel(
            logging.DEBUG if include_debug_in_training or debug_to_file else logging.INFO
        )
        if log_to_driver:
            clear_level_file_handlers(logger_ref)
            ensure_console_handler(
                logger_ref, handler_role, rank, level=console_level, use_current_stderr=True
            )
            if debug_to_file:
                ensure_debug_file_handler(logger_ref, log_dir, handler_role, rank)
        else:
            install_level_handlers(
                logger_ref, log_dir, handler_role, rank, include_debug_in_training, debug_to_file
            )
    infer_engine_log_path = None
    if resolve_capture_infer_engine_log():
        infer_engine_log_path = get_infer_engine_log_path(
            log_dir,
            backend=backend,
            role=role,
            engine_idx=engine_idx,
            rank=rank,
            rm_idx=rm_idx,
        )
    os.environ["GPATCH_ENGINE_LOG_FILE"] = infer_engine_log_path or ""
    os.environ["GPATCH_ENGINE_LOG_DIR"] = (
        os.path.dirname(infer_engine_log_path) if infer_engine_log_path else log_dir
    )
    os.environ["GPATCH_ENGINE_ROLE"] = role
    # Route the third-party engine's Python logging into infer_engine_log so
    # that metrics like throughput (tokens/s) appear alongside other engine logs.
    if backend == "vllm" and infer_engine_log_path:
        install_engine_file_handler(backend, infer_engine_log_path)
    return infer_engine_log_path


def install_engine_file_handler(backend: str, log_path: str) -> None:
    """Add a FileHandler so the engine's Python logging goes to ``infer_engine_log``.

    Captures metrics (e.g. throughput) that the engine logs via Python
    logging rather than stdout.
    """
    logger_name = backend  # "vllm" or "sglang"
    engine_logger = logging.getLogger(logger_name)
    for handler in list(engine_logger.handlers):
        if getattr(handler, "_gpatch_engine_log", None) is None:
            continue
        engine_logger.removeHandler(handler)
        handler.close()
    Path(os.path.dirname(log_path) or ".").mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="a")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    handler._gpatch_engine_log = log_path
    engine_logger.addHandler(handler)


def get_infer_engine_log_dir(log_dir: Optional[str] = None) -> str:
    if log_dir is None:
        log_dir = os.environ.get(TASK_LOG_DIR_ENV) or derive_task_log_dir()
    return os.path.join(log_dir, "infer_engine_log")


def write_infer_engine_log_marker(ppo_step: int, phase: str = "begin") -> None:
    """Write a step marker line to infer engine log file(s).

    Inside a sampler actor (``GPATCH_ENGINE_LOG_FILE`` set) writes to that
    single file; otherwise broadcasts to every ``*.log`` under
    ``infer_engine_log/``.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    marker = f"\n{'=' * 60}\n[ppo_step={ppo_step}] {phase} - {timestamp}\n{'=' * 60}\n"

    log_file = os.environ.get("GPATCH_ENGINE_LOG_FILE")
    if log_file:
        _write_marker_to_file(log_file, marker)
        return

    log_dir = os.environ.get(TASK_LOG_DIR_ENV)
    if not log_dir:
        return
    engine_log_dir = get_infer_engine_log_dir(log_dir)
    if not os.path.isdir(engine_log_dir):
        return
    targets = [n for n in os.listdir(engine_log_dir) if n.endswith(".log")]
    for name in targets:
        _write_marker_to_file(os.path.join(engine_log_dir, name), marker)


def _write_marker_to_file(path: str, marker: str) -> None:
    try:
        with open(path, "a") as f:
            f.write(marker)
            f.flush()
    except OSError:
        pass


def get_infer_engine_log_path(
    log_dir: Optional[str] = None,
    backend: str = "infer_engine",
    role: str = "infer_engine",
    engine_idx: Optional[int] = None,
    rank: Optional[int] = None,
    rm_idx: Optional[int] = None,
) -> str:
    backend = sanitize_logger_name(backend)
    role = sanitize_logger_name(normalize_role_alias(role))
    if rm_idx is not None:
        role = f"{role}{rm_idx}"
    engine = "none" if engine_idx is None else str(engine_idx)
    return os.path.join(
        get_infer_engine_log_dir(log_dir),
        f"{backend}_{role}_engine{engine}_rank{rank_to_str(rank)}.log",
    )


def redirect_stdio_to_level_logs(
    log_dir: Optional[str] = None,
    role: str = "worker",
    rank: Optional[int] = None,
    tee_to_console: bool = True,
) -> None:
    """Tee stderr error patterns into training.log."""
    global stdio_redirected
    if stdio_redirected:
        return
    if os.environ.get(LOG_STDIO_ENV, "1") in {"0", "false", "False"}:
        return
    if rank is None:
        rank = get_current_rank()
    if log_dir is None:
        log_dir = os.environ.get(TASK_LOG_DIR_ENV) or derive_task_log_dir()

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    training_log_fd = os.open(
        os.path.join(log_dir, "training.log"),
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o644,
    )
    sys.stdout = TeeStream(
        sys.__stdout__ if tee_to_console else None,
        training_log_fd,
        "stdout",
        role,
        rank,
    )
    sys.stderr = TeeStream(
        sys.__stderr__ if tee_to_console else None,
        training_log_fd,
        "stderr",
        role,
        rank,
    )
    stdio_redirected = True


@contextmanager
def redirect_stdio_fds_to_file(path: str):
    """Temporarily redirect stdout/stderr fds so child processes inherit the file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    target_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        os.dup2(target_fd, 1)
        os.dup2(target_fd, 2)
        yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(target_fd)


def get_current_rank() -> Optional[int]:
    for key in ("RANK", "LOCAL_RANK"):
        val = os.environ.get(key)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                return None
    return None


def resolve_log_level(config=None) -> int:
    value = first_non_empty(
        nested_get(config, "report", "log_level"),
        os.environ.get(LOG_LEVEL_ENV),
    )
    if value is None:
        return logging.INFO
    if isinstance(value, int):
        return value

    level_name = str(value).strip().upper()
    level = logging.getLevelName(level_name)
    if not isinstance(level, int):
        raise ValueError(f"invalid log_level={value!r}")
    return level


def resolve_debug_log_to_file(config=None) -> bool:
    value = first_non_empty(
        nested_get(config, "report", "debug_log_to_file"),
        os.environ.get(DEBUG_LOG_TO_FILE_ENV),
    )
    return parse_bool(value, default=False)


def resolve_capture_infer_engine_log(config=None) -> bool:
    value = first_non_empty(
        nested_get(config, "report", "capture_infer_engine_log"),
        os.environ.get(CAPTURE_INFER_ENGINE_LOG_ENV),
    )
    return parse_bool(value, default=True)


def resolve_log_to_driver(config=None) -> bool:
    value = first_non_empty(
        nested_get(config, "report", "log_to_driver"),
        os.environ.get(LOG_TO_DRIVER_ENV),
    )
    return parse_bool(value, default=False)


def install_level_handlers(
    logger: logging.Logger,
    log_dir: str,
    role: str,
    rank: Optional[int],
    include_debug_in_training: bool = False,
    debug_to_file: bool = False,
) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    specs = [
        (
            "training",
            os.path.join(log_dir, "training.log"),
            logging.DEBUG if include_debug_in_training else logging.INFO,
            TrainingLevelFilter(include_debug_in_training),
        ),
    ]
    if debug_to_file:
        ensure_debug_file_handler(logger, log_dir, role, rank)
    for name, path, level, level_filter in specs:
        ensure_level_file_handler(logger, name, path, level, level_filter, role, rank)


def ensure_level_file_handler(
    logger: logging.Logger,
    name: str,
    file_path: str,
    level: int,
    level_filter: logging.Filter,
    role: str,
    rank: Optional[int],
) -> None:
    abs_path = os.path.abspath(file_path)
    for handler in logger.handlers:
        if getattr(handler, "gpatch_level_handler", None) == name:
            if getattr(handler, "gpatch_file_path", None) == abs_path:
                handler.setLevel(level)
                reset_handler_filters(handler, role, rank, level_filter)
                return
            logger.removeHandler(handler)
            handler.close()
            break
    handler = AtomicRecordHandler(abs_path)
    handler.setLevel(level)
    handler.setFormatter(make_formatter())
    handler.gpatch_level_handler = name
    handler.gpatch_file_path = abs_path
    reset_handler_filters(handler, role, rank, level_filter)
    logger.addHandler(handler)


def ensure_debug_file_handler(
    logger: logging.Logger,
    log_dir: str,
    role: str,
    rank: Optional[int],
) -> None:
    ensure_level_file_handler(
        logger,
        "debug",
        get_debug_log_path(log_dir, role, rank),
        logging.DEBUG,
        DebugLevelFilter(),
        role,
        rank,
    )


def get_debug_log_path(log_dir: str, role: str, rank: Optional[int]) -> str:
    filename = (
        f"debug_{sanitize_logger_name(normalize_role_alias(role))}_"
        f"rank{rank_to_str(rank)}_pid{os.getpid()}.log"
    )
    return os.path.join(log_dir, DEBUG_LOG_DIRNAME, filename)


def clear_level_file_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if getattr(handler, "gpatch_level_handler", None) is not None:
            logger.removeHandler(handler)
            handler.close()


def clear_plain_console_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if getattr(handler, "gpatch_console", False):
            continue
        if getattr(handler, "gpatch_level_handler", None) is not None:
            continue
        if isinstance(handler, logging.FileHandler):
            continue
        if not isinstance(handler, logging.StreamHandler):
            continue
        logger.removeHandler(handler)
        handler.close()


def ensure_console_handler(
    logger: logging.Logger,
    role: str,
    rank: Optional[int],
    level: int = logging.INFO,
    use_current_stderr: bool = False,
) -> None:
    stream = get_console_stream(use_current_stderr)
    for handler in logger.handlers:
        if getattr(handler, "gpatch_console", False):
            handler.setLevel(level)
            if getattr(handler, "stream", None) is not stream:
                handler.setStream(stream)
            reset_handler_filters(handler, role, rank)
            return
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(make_formatter())
    handler.gpatch_console = True
    reset_handler_filters(handler, role, rank)
    logger.addHandler(handler)


def get_console_stream(use_current_stderr: bool = False):
    if use_current_stderr and not isinstance(sys.stderr, TeeStream):
        return sys.stderr
    return sys.__stderr__


def reset_handler_filters(
    handler: logging.Handler,
    role: str,
    rank: Optional[int],
    level_filter: Optional[logging.Filter] = None,
) -> None:
    handler.filters = [
        f for f in handler.filters if not isinstance(
            f,
            (
                GpatchContextFilter,
                TrainingLevelFilter,
                DebugLevelFilter,
            ),
        )
    ]
    handler.addFilter(GpatchContextFilter(role, rank))
    if level_filter is not None:
        handler.addFilter(level_filter)


def make_formatter() -> logging.Formatter:
    return logging.Formatter(
        "%(asctime)s - %(levelname)s - role=%(gpatch_role)s - "
        "rank=%(gpatch_rank)s - pid=%(gpatch_pid)s - node=%(gpatch_node)s - %(message)s"
    )


def nested_get(obj, *attrs):
    cur = obj
    for attr in attrs:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(attr)
        else:
            cur = getattr(cur, attr, None)
    return cur


def first_non_empty(*values):
    for val in values:
        if val not in (None, ""):
            return val
    return None


def parse_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def sanitize_path_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("._") or "default"


def sanitize_logger_name(value: str) -> str:
    return sanitize_path_component(value).replace("-", "_")


def rank_to_str(rank: Optional[int]) -> str:
    return "none" if rank is None else str(rank)


def get_node_ip() -> str:
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "unknown"


def normalize_role_alias(role: str) -> str:
    return role.replace("-", "_")


def install_excepthook(logger: logging.Logger) -> None:
    """Install sys.excepthook that logs uncaught exceptions to training.log."""
    def _excepthook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))

    sys.excepthook = _excepthook


def install_compat_logger(logger: logging.Logger) -> None:
    set_default_logger(logger)
