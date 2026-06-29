import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from gpatch_v4.custom.registry import (
    cleanup_custom_module_registry,
    get_custom_module_registry_path,
)
from gpatch_v4.utils.common_utils import import_fn_from_path


def test_dynamic_module_registry_resolves_in_spawned_process(tmp_path, monkeypatch):
    monkeypatch.delenv("GPATCH_CUSTOM_MODULE_REGISTRY_PATH", raising=False)
    py_path = tmp_path / "user_dataset.py"
    py_path.write_text(
        "def marker():\n"
        "    return 'registry-ok'\n",
        encoding="utf-8",
    )

    mod_hash = hashlib.md5(str(py_path).encode("utf-8")).hexdigest()
    full_mod_name = f"gpatch_v4.custom.{mod_hash}"

    try:
        marker = import_fn_from_path(str(py_path), "marker")
        assert marker() == "registry-ok"

        registry_path = get_custom_module_registry_path(create=False)
        assert registry_path is not None
        assert registry_path.startswith("/tmp/gpatch_v4_custom_registry/")
        with open(registry_path, "r") as f:
            registry = json.load(f)
        assert registry[mod_hash] == str(py_path)

        script = (
            "import importlib\n"
            f"module = importlib.import_module({full_mod_name!r})\n"
            "assert module.marker() == 'registry-ok'\n"
        )
        gcore_dev = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(gcore_dev), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(gcore_dev)]
        )
        subprocess.run([sys.executable, "-c", script], env=env, check=True)
    finally:
        sys.modules.pop(full_mod_name, None)
        cleanup_custom_module_registry()
        registry_path = os.environ.get("GPATCH_CUSTOM_MODULE_REGISTRY_PATH")
        assert registry_path is None or not os.path.exists(registry_path)
