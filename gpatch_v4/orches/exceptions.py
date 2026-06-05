"""Orchestration exception types.

Re-exports backend-specific exceptions so that code outside
``orches/`` can catch actor/task errors without importing the
orchestration backend (e.g. Ray) directly.
"""

import ray.exceptions

RayActorError = ray.exceptions.RayActorError
RayTaskError = ray.exceptions.RayTaskError
