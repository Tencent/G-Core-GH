"""FrozenLake text env for gcore-dev agentic GRPO (ported from roll_webshop)."""
from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv as GymFrozenLakeEnv


class _GemEnvStub:
    def reset(self, seed=None):
        pass


try:
    from gem import Env
except ImportError:
    Env = _GemEnvStub

from gpatch_v4.agentic.env.frozenlake.frozenlake_utils import generate_random_map
from gpatch_v4.agentic.env.frozenlake.parse_action_utils import (
    default_parser_action_func,
)
from gpatch_v4.agentic.env.frozenlake.random_utils import all_seed


class FrozenLakeEnv(Env, GymFrozenLakeEnv):
    def __init__(
        self,
        render_mode: str = "text",
        size: int = 4,
        p: float = 0.8,
        is_slippery: bool = True,
        map_seed: Optional[int] = None,
        max_steps: int = 20,
        grid_lookup=None,
        grid_vocab=None,
        map_lookup=None,
        action_lookup=None,
        env_instruction=None,
        format_penalty: float = 0.0,
        action_pattern: str = r"<answer>(.*?)</answer>",
        special_token_list=("<|im_start|>", "<|im_end|>"),
        **kwargs,
    ):
        self.GRID_LOOKUP = {0: "P", 1: "_", 2: "O", 3: "G", 4: "X", 5: "√"}
        self.GRID_VOCAB = {
            "P": "player",
            "_": "empty",
            "O": "hole",
            "G": "goal",
            "X": "player in hole",
            "√": "player on goal",
        }
        self.ACTION_LOOKUP = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
        self.MAP_LOOKUP = {b"P": 0, b"F": 1, b"H": 2, b"G": 3}
        self.env_instruction = (
            "You are solving the FrozenLake puzzle. "
            "Forbid the whole and go to the target. "
            "You may move to the unintended direction due to the slippery ice. "
            "The answer must be one of action in a turn, format is <answer>Right</answer>"
        )
        if grid_lookup is not None:
            self.GRID_LOOKUP = grid_lookup
        if grid_vocab is not None:
            self.GRID_VOCAB = grid_vocab
        if action_lookup is not None:
            self.ACTION_LOOKUP = action_lookup
        if env_instruction is not None:
            self.env_instruction = env_instruction
        if map_lookup is not None:
            self.MAP_LOOKUP = map_lookup
        self.size = size
        self.p = p
        self.is_slippery = is_slippery
        self.map_seed = map_seed
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.format_penalty = format_penalty
        self.action_pattern = action_pattern
        self.special_token_list = special_token_list

        random_map = generate_random_map(size=self.size, p=self.p, seed=map_seed)
        GymFrozenLakeEnv.__init__(
            self,
            desc=random_map,
            is_slippery=is_slippery,
            render_mode=self.render_mode,
            **kwargs,
        )
        self.step_count = 0

    def get_instructions(self) -> str:
        grid_vocab_str = "\nThe meaning of each symbol in the state is:\n" + ", ".join(
            [f"{k}: {v}" for k, v in self.GRID_VOCAB.items()]
        )
        action_lookup_str = "\nYour available actions are:\n" + ", ".join(
            [f"{v}" for k, v in self.ACTION_LOOKUP.items()]
        )
        return self.env_instruction + grid_vocab_str + action_lookup_str

    def reset(
        self,
        seed: int = 42,
        ppo_step: int = 0,
        data: Optional[Dict[str, Any]] = None,
    ):
        Env.reset(self, seed)
        self.step_count = 0
        map_seed = seed
        if data is not None and "map_seed" in data:
            ms = data["map_seed"]
            map_seed = ms[0] if isinstance(ms, list) else ms
        try:
            with all_seed(map_seed):
                random_map = generate_random_map(size=self.size, p=self.p, seed=map_seed)
                GymFrozenLakeEnv.__init__(
                    self,
                    desc=random_map,
                    is_slippery=self.is_slippery,
                    render_mode=self.render_mode,
                )
                GymFrozenLakeEnv.reset(self, seed=map_seed)
                observation = self.render(mode=self.render_mode)
                info = {"env_instruction": self.get_instructions()}
                return observation, info, False
        except (RuntimeError, RuntimeWarning):
            next_seed = abs(hash(str(seed))) % (2**32) if seed is not None else None
            return self.reset(next_seed, ppo_step=ppo_step, data=data)

    def step(
        self,
        action: str,
        tool_call_parse_result=None,
        reasoning_parse_result=None,
    ) -> Tuple[Any, float, bool, bool, Dict]:
        del tool_call_parse_result, reasoning_parse_result
        metrics_agg_mode = {
            "action_is_effective": "mean",
            "action_is_valid": "mean",
            "success": "last",
            "format_penalty": "mean",
        }

        self.step_count += 1
        action_info = self.parse_action(action)
        if action_info["action"] is None:
            next_obs = self.render()
            action_desc = f"At turn {self.step_count}, You did not provide a valid action."
            reward = self.format_penalty
            metrics = {
                "action_is_effective": False,
                "action_is_valid": False,
                "success": self.desc[self.player_pos] == b"G",
                "format_penalty": self.format_penalty,
            }
            info = {
                "metrics": metrics,
                "metrics_agg_mode": metrics_agg_mode,
                "action_desc": action_desc,
            }
            info.update(action_info)
            return next_obs, reward, False, False, info

        prev_pos = int(self.s)
        _, reward, terminated, truncated, _ = GymFrozenLakeEnv.step(self, action_info["action"])
        next_obs = self.render()

        action_effective = prev_pos != int(self.s)
        if not action_effective:
            action_desc = (
                f"At turn {self.step_count}, you tried to move {action_info['action_content']}, "
                "which is not effective yet."
            )
        else:
            action_desc = (
                f"At turn {self.step_count}, you moved {action_info['action_content']}, "
                "which is effective."
            )

        metrics = {
            "action_is_effective": action_effective,
            "action_is_valid": True,
            "success": self.desc[self.player_pos] == b"G",
            "format_penalty": self.format_penalty,
        }
        info = {
            "metrics": metrics,
            "metrics_agg_mode": metrics_agg_mode,
            "action_desc": action_desc,
        }
        info.update(action_info)
        if terminated:
            if not metrics["success"] and self.step_count >= self.max_steps:
                truncated = True
        return next_obs, reward, terminated, truncated, info

    def parse_action(self, text):
        return default_parser_action_func(
            text, self.action_pattern, self.ACTION_LOOKUP, self.special_token_list
        )

    def render(self, mode=None):
        if not mode:
            mode = self.render_mode
        if mode == "text":
            room = self.desc.copy()
            room = np.where(room == b"S", b"F", room)
            room[self.player_pos] = b"P"
            room = np.vectorize(lambda x: self.MAP_LOOKUP[x])(room)
            room[self.player_pos] = (
                4 if self.desc[self.player_pos] == b"H" else
                5 if self.desc[self.player_pos] == b"G" else 0
            )
            return "\n".join(
                "".join(self.GRID_LOOKUP.get(cell, "?") for cell in row) for row in room
            )
        if mode == "rgb_array":
            return self._render_gui("rgb_array")
        raise ValueError(f"Invalid mode: {self.render_mode}")

    @property
    def player_pos(self):
        return (self.s // self.ncol, self.s % self.ncol)

    def close(self):
        super(FrozenLakeEnv, self).close()
