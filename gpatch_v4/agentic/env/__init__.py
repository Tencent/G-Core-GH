from __future__ import annotations

from typing import Any

import gem


def register_custom_env_for_agentic(agentic_cfg: Any) -> None:
    """Register every ``gem`` env whose template sets ``custom_env_cls``.

    Iterates ``agentic_cfg.resolved_env_templates()`` so both single-env
    (``env_cfg_template``) and multi-env (``env_cfg_templates``) runs register
    all of their custom envs. Each template's ``custom_env_cls`` is registered
    under its own ``env_type`` for ``gem.make``.
    """
    for tmpl in agentic_cfg.resolved_env_templates():
        custom = tmpl.custom_env_cls
        if custom == "":
            continue
        env_type = tmpl.env_type
        if not env_type:
            raise ValueError("custom_env_cls is set but env_cfg_template.env_type is empty")
        if ":" not in custom:
            raise ValueError(f"custom_env_cls must be 'module.path:ClassName', got {custom!r}")
        gem.register(env_type, entry_point=custom)
