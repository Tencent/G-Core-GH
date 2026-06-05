from __future__ import annotations

from typing import Any

import gem


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
