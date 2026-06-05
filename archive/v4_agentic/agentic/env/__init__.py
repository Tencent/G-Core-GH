"""
base agentic codes reference: https://github.com/RAGEN-AI/RAGEN

Custom env: set ``env_cfg_template.custom_env_cls`` (``module:Class``) and ``env_cfg_template.env_type``;
``EnvironmentWorker.initialize`` calls ``register_custom_env_for_agentic`` before ``gem.make``.
"""
from __future__ import annotations

from contextlib import suppress
from typing import Any

with suppress(ImportError):
    import gem

    gem.register("sokoban", entry_point="gpatch_v4.agentic.env.sokoban:SokobanEnv")
    gem.register("miniprogram", entry_point="gpatch_v4.agentic.env.miniprogram:MiniprogramEnv")
    gem.register(
        "interactive_cli",
        entry_point="gpatch_v4.agentic.env.interactive_cli.interactive_cli_env:InteractiveCliEnv",
    )


def register_custom_env_for_agentic(agentic_cfg: Any) -> None:
    """Register ``gem`` env when ``env_cfg_template.custom_env_cls`` is set (uses ``env_cfg_template.env_type``)."""
    if agentic_cfg.env_cfg_template.custom_env_cls == "":
        return
    env_type = agentic_cfg.env_cfg_template.env_type
    custom = agentic_cfg.env_cfg_template.custom_env_cls
    if not env_type:
        raise ValueError("custom_env_cls is set but env_cfg_template.env_type is empty")
    if ":" not in custom:
        raise ValueError(f"custom_env_cls must be 'module.path:ClassName', got {custom!r}")
    gem.register(env_type, entry_point=custom)
