from typing import Dict, List
import json
from dataclasses import asdict
import numpy as np
import torch

def _rollout_cache_json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return str(obj)

class EnvManagerStrMixin:
        
    def __str__(self) -> str:
        group_id = getattr(self, "group_id", self.env_config["group_id"])
        env_id = getattr(self, "env_id", self.env_config["env_id"])
        env_type = getattr(self, "env_type", self.env_config["env_type"])
        episode_id = getattr(self, "episode_id", 0)
        current_step = getattr(self, "current_step", -1)
        rollout_cache = (
            asdict(self.rollout_cache) if self.rollout_cache is not None else None
        )
        payload = {
            "group_id": group_id,
            "env_id": env_id,
            "env_type": env_type,
            "episode_id": episode_id,
            "current_step": current_step,
            "rollout_cache": rollout_cache,
        }
        return json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=_rollout_cache_json_default,
        )