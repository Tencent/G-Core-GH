import importlib.util
import logging
import unittest

import ray
from hydra import compose, initialize

from gpatch_v4.configs.utils import merge_hydra_config


def load_config(config_name, ConfigClass):
    with initialize(config_path="configs/test_yaml", version_base=None):
        config = compose(config_name=config_name)
        config = merge_hydra_config(ConfigClass, config)
    return config


# ``unittest.skipUnless`` decorator that activates whenever the current
# container/image does not ship ``vllm``. Tests exercising the vllm backend
# still run unchanged on any image that does have vllm installed, and are
# transparently skipped otherwise -- which matches how we sometimes swap
# the docker image to one without vllm for sglang-only regression runs.
_VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None
requires_vllm = unittest.skipUnless(_VLLM_AVAILABLE, "vllm not installed in this image")

_SGLANG_AVAILABLE = importlib.util.find_spec("sglang") is not None
requires_sglang = unittest.skipUnless(_SGLANG_AVAILABLE, "sglang not installed in this image")

# ---------------------------------------------------------------------------
# kill_all_actors_and_shutdown_ray()
#
# Test tearDown helper that prevents sglang subprocess leaks between tests.
#
# Design (minimal, three steps):
#
#   1. For every named actor visible via ``ray.util.list_named_actors``,
#      invoke ``actor.shutdown.remote()`` with a bounded timeout. The
#      actor's ``shutdown`` (see ``BaseActor.shutdown`` for the default
#      no-op and overrides in ``GrpoSamplerActor`` / ``GrpoGenRmActor`` /
#      ``T2iGrpoGenRmActor`` for sglang reaping via ``kill_process_tree``)
#      is responsible for releasing GPU memory and reaping child processes
#      while the ray worker is still alive.
#   2. ``ray.kill(actor, no_restart=True)`` every named actor so its name
#      is released for the next test (fixing the bug at
#      tests/test_gpatch_v4/test_failure_recovery.py:416-427 which silently
#      killed nothing due to a ``TypeError`` swallowed by a broad except).
#   3. ``ray.shutdown()`` on this driver.
#
# Everything is non-raising and idempotent.
# ---------------------------------------------------------------------------

_LOG = logging.getLogger(__name__)


def _list_actor_handles():
    """Return ``[(info, handle), ...]`` for every named actor visible to ray.

    Fixes the bug in ``tests/test_gpatch_v4/test_failure_recovery.py:416-427``
    which passed the dict returned by ``list_named_actors(all_namespaces=True)``
    as a positional ``name`` into ``get_actor``, raising ``TypeError`` that
    was silently swallowed → no actor was actually killed.
    """
    out = []
    try:
        if not ray.is_initialized():
            return out
        for info in ray.util.list_named_actors(all_namespaces=True):
            try:
                if isinstance(info, dict):
                    name = info["name"]
                    ns = info.get("namespace")
                else:
                    name = info
                    ns = None
                handle = ray.get_actor(name, namespace=ns) if ns else ray.get_actor(name)
                out.append((info, handle))
            except Exception as e:
                _LOG.warning("get_actor(%r) failed: %s", info, e)
    except Exception as e:
        _LOG.warning("list_named_actors failed: %s", e)
    return out


def kill_all_actors_and_shutdown_ray(grace_seconds: float = 30.0) -> None:
    """Idempotent, non-raising tearDown helper.

    Parameters
    ----------
    grace_seconds : float, optional
        ``ray.get`` timeout on each ``actor.shutdown.remote()`` call.
        Defaults to 30s because sglang's own ``kill_process_tree`` can
        take a few seconds on large TP groups. Individual timeouts are
        isolated: a hung actor does not block cleanup of the rest.
    """
    handles = _list_actor_handles()

    # Step 1 — graceful shutdown: each actor reaps its own subprocesses.
    for info, handle in handles:
        method = getattr(handle, "shutdown", None)
        if method is None:
            continue  # actor class has no shutdown — ray will clean up
        try:
            ray.get(method.remote(), timeout=grace_seconds)
        except (AttributeError, ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError) as e:
            _LOG.warning("graceful shutdown for %r failed: %s", info, e)
        except Exception as e:  # pragma: no cover -- best effort
            _LOG.warning("graceful shutdown for %r unexpected: %s", info, e)

    # Step 2 — ray.kill each named actor (releases the name for next test,
    # and triggers ray-level cleanup for actors that had no shutdown override).
    for info, handle in handles:
        try:
            ray.kill(handle, no_restart=True)
        except Exception as e:  # pragma: no cover -- best effort
            _LOG.warning("ray.kill(%r) failed: %s", info, e)

    # Step 3 — shutdown ray on this driver.
    try:
        if ray.is_initialized():
            ray.shutdown()
    except Exception as e:  # pragma: no cover -- best effort
        _LOG.warning("ray.shutdown() failed: %s", e)
