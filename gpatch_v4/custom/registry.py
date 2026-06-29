import atexit
import json
import os
import re
import tempfile
import threading
import uuid
from typing import Optional

CUSTOM_MODULE_REGISTRY_ENV = "GPATCH_CUSTOM_MODULE_REGISTRY_PATH"
CUSTOM_MODULE_REGISTRY_DIR = "/tmp/gpatch_v4_custom_registry"
REGISTRY_FILE_RE = re.compile(r"^registry_(\d+)_(\d+)_[0-9a-f]{32}\.json$")

WRITE_LOCK = threading.Lock()
OWNED_REGISTRY_PATH: Optional[str] = None


def get_custom_module_registry_path(create: bool = True) -> Optional[str]:
    registry_path = os.environ.get(CUSTOM_MODULE_REGISTRY_ENV)
    if registry_path:
        return registry_path
    if not create:
        return None

    global OWNED_REGISTRY_PATH

    os.makedirs(CUSTOM_MODULE_REGISTRY_DIR, exist_ok=True)
    cleanup_stale_custom_module_registries()
    registry_path = os.path.join(
        CUSTOM_MODULE_REGISTRY_DIR,
        f"registry_{os.getuid()}_{os.getpid()}_{uuid.uuid4().hex}.json",
    )
    os.environ[CUSTOM_MODULE_REGISTRY_ENV] = registry_path
    OWNED_REGISTRY_PATH = registry_path
    return registry_path


def load_custom_module_registry() -> dict[str, str]:
    registry_path = get_custom_module_registry_path(create=False)
    if registry_path is None or not os.path.exists(registry_path):
        return {}
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    if not isinstance(registry, dict):
        raise ValueError(f"invalid custom module registry format: {registry_path}")
    return registry


def register_custom_module(mod_name: str, py_path: str) -> None:
    registry_path = get_custom_module_registry_path(create=True)
    assert registry_path is not None
    abs_py_path = os.path.abspath(py_path)

    with WRITE_LOCK:
        registry = load_custom_module_registry()
        if mod_name in registry:
            if registry[mod_name] != abs_py_path:
                raise ValueError(
                    f"custom module registry collision: {mod_name} maps to "
                    f"{registry[mod_name]}, not {abs_py_path}"
                )
            return
        registry[mod_name] = abs_py_path
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".registry.",
            suffix=".json",
            dir=os.path.dirname(registry_path),
        )
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(registry, f, sort_keys=True)
            os.replace(tmp_path, registry_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise


def cleanup_custom_module_registry() -> None:
    registry_path = OWNED_REGISTRY_PATH
    if registry_path is not None:
        try:
            os.unlink(registry_path)
        except FileNotFoundError:
            pass
        if os.environ.get(CUSTOM_MODULE_REGISTRY_ENV) == registry_path:
            os.environ.pop(CUSTOM_MODULE_REGISTRY_ENV, None)
    cleanup_stale_custom_module_registries()


def cleanup_stale_custom_module_registries() -> None:
    try:
        filenames = os.listdir(CUSTOM_MODULE_REGISTRY_DIR)
    except FileNotFoundError:
        return

    uid = os.getuid()
    for filename in filenames:
        match = REGISTRY_FILE_RE.match(filename)
        if match is None:
            continue
        file_uid = int(match.group(1))
        if file_uid != uid:
            continue
        pid = int(match.group(2))
        if os.path.exists(f"/proc/{pid}"):
            continue
        path = os.path.join(CUSTOM_MODULE_REGISTRY_DIR, filename)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


atexit.register(cleanup_custom_module_registry)
