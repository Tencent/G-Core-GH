# This package serves as a namespace for dynamically imported modules
# registered via `import_mod_from_path()`. It must exist on disk so that
# forkserver child processes can resolve ``gpatch_v4.custom.<hash>`` modules
# during pickle deserialization.

import importlib
import importlib.abc
import importlib.util
import os
import sys

from gpatch_v4.custom.registry import load_custom_module_registry


class _CustomModuleFinder(importlib.abc.MetaPathFinder):
    """Finder that resolves ``gpatch_v4.custom.<hash>`` modules from the registry."""
    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith('gpatch_v4.custom.'):
            return None
        short_name = fullname[len('gpatch_v4.custom.'):]
        registry = self._load_registry()
        if short_name not in registry:
            return None
        py_path = registry[short_name]
        if not os.path.exists(py_path):
            return None
        return importlib.util.spec_from_file_location(fullname, py_path)

    @staticmethod
    def _load_registry():
        return load_custom_module_registry()


# Register the finder so that child processes (e.g. forkserver workers)
# can import dynamically-registered modules by name.
sys.meta_path.append(_CustomModuleFinder())
