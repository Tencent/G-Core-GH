from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

from gpatch_v4.agentic.env import gem
from gpatch_v4.agentic.proto import DataProto


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




class BaseEnvManager:
    def __init__(self, *args, **kwargs):
        self.current_step = -1
        self.running = False
        self.env: gem.Env

    @abstractmethod
    def run_rollout_loop(self, data: DataProto):
        """
        1. Each time run_rollout_loop is called,
           it will continuously play episodes until it receives a command that data collection is complete.
           The seed needs to be reset to ensure consistency across all groups.
           episode_id is reset to 0.

        Seed update logic:
           group_seed = base_seed + group_id
           episode_seed = group_seed + episode_id

        trajectory_id: f"{group_id}_{episode_id}_{episode_seed}"
        """
        pass

    def reset(self) -> RolloutCache:
        pass

    def step(self, llm_output: DataProto) -> RolloutCache:
        pass

    def make_decision(self, rollout_cache: RolloutCache) -> DataProto:
        pass

    def format_messages(self, history: List[Dict]) -> DataProto:
        pass

    def formulate_rollouts(self, rollout_cache: RolloutCache) -> DataProto:
        pass

    def update_step(self, global_step):
        self.current_step = global_step

    def stop(self):
        self.running = False
