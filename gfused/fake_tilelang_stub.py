"""Minimal tilelang stubs so kernel modules can be imported without tilelang.

@tilelang.jit decorated functions become no-ops that raise RuntimeError
when actually called. Module-level constants (T.bfloat16 etc.) resolve
to sentinels instead of crashing.
"""

import functools


class _HashableSentinel:
    """Hashable placeholder for PassConfigKey enum values."""
    def __init__(self, name: str = ""):
        self._name = name

    def __repr__(self):
        return f"<stub:{self._name}>"

    def __hash__(self):
        return hash(self._name)

    def __eq__(self, other):
        return isinstance(other, _HashableSentinel) and self._name == other._name


class _PassConfigKeyStub:
    def __getattr__(self, name: str):
        return _HashableSentinel(name)


def _jit(_fn=None, **_kwargs):
    """No-op replacement for ``tilelang.jit``.

    Handles both ``@jit`` and ``@jit(...)`` forms.
    The wrapped function raises at call time.
    """
    def _decorator(fn):
        @functools.wraps(fn)
        def _wrapper(*args, **kwargs):
            raise RuntimeError(f"tilelang is not installed; cannot call {fn.__qualname__}")

        return _wrapper

    if _fn is not None:
        return _decorator(_fn)
    return _decorator


class TilelangStub:
    jit = staticmethod(_jit)
    PassConfigKey = _PassConfigKeyStub()


class LanguageStub:
    bfloat16 = _HashableSentinel("bfloat16")
    float16 = _HashableSentinel("float16")
    float32 = _HashableSentinel("float32")
    float = _HashableSentinel("float")
    int32 = _HashableSentinel("int32")

    def __getattr__(self, name: str):
        return _HashableSentinel(name)


tilelang_stub = TilelangStub()
language_stub = LanguageStub()
print(f"Warning: import tilelang error， fallback to fake_tilelang_stub")
