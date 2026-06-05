import json
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch

from gpatch_v4.agentic.env import gem


@dataclass
class RolloutCache:
    env_id: int
    group_id: int
    tag: str

    history: List[Dict] = field(
        default_factory=list
    )  # keys: [state, actions_left, reward, penalty, llm_response, metrics], a dict save each step info
    frames: List = field(default_factory=list)

    truncated: bool = False
    terminated: bool = False
    step: int = 0
    terminated_reason: str = None

    def fork(self, step):
        assert step <= self.step
        forked = type(self)(self.env_id, self.group_id, tag=self.tag)
        forked.history = self.history[:step]
        if self.frames:
            forked.frames = self.frames[step]
        forked.truncated = False
        forked.terminated = False
        forked.step = step
        forked.terminated_reason = self.terminated_reason
        return forked


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
        rollout_cache = (asdict(self.rollout_cache) if self.rollout_cache is not None else None)
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
