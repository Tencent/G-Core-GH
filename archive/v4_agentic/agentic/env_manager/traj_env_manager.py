"""Simplified TrajEnvManager for async rollout architecture.

Runs a single trajectory and returns a rollout_batch dict directly,
without DataProto or multi-episode management.  Designed to be driven
by ``EnvAgentLoopActor`` (e.g. ``asyncio.to_thread``).
"""

from contextlib import suppress
from typing import Any, Dict, List, Optional

with suppress(ImportError):
    import gem

import torch
from transformers import PreTrainedTokenizer

from gpatch_v4.agentic.env.tool_env_wrapper import ToolEnvWrapper
from gpatch_v4.agentic.env_manager.base_env_manager import RolloutCache
from gpatch_v4.agentic.env_manager.token_mask_utils import (
    compute_conversation_end_token_id,
    custom_apply_chat_template,
)
from gpatch_v4.agentic.llm_proxy import create_llm_proxy
from gpatch_v4.agentic.utils import LoggerAdaptor
from gpatch_v4.client import SamplerClient
from gpatch_v4.configs.config import AgenticRlConfig
from gpatch_v4.utils.constants import GenerateStopReason
from gpatch_v4.utils.str_utils import contains_renderable_field
from gpatch_v4.utils import log


class TrajEnvManager:
    """Lightweight env manager that runs exactly one trajectory.

    Parameters
    ----------
    rl_config : AgenticRlConfig
        Top-level RL training configuration.
    env_config : dict
        Per-env configuration (env_type, config, env_id, group_id, tag,
        max_steps, etc.).
    tokenizer : PreTrainedTokenizer
        Tokenizer instance (should be a dedicated copy per manager for
        thread safety).
    sampler_client : SamplerClient
        Shared sampler RPC client.
    """

    def __init__(
        self,
        rl_config: AgenticRlConfig,
        env_config: Dict,
        tokenizer: PreTrainedTokenizer,
        sampler_client: SamplerClient,
    ):
        self.rl_config = rl_config
        self.env_config = env_config
        self.tokenizer = tokenizer
        self.sampler_client = sampler_client
        self.logger = LoggerAdaptor()
        self.rollout_cache: Optional[RolloutCache] = None
        self.env = None
        self.llm_proxy = None

        agentic_config = rl_config.training.agentic
        self.cfg_template = agentic_config.env_cfg_template
        self.agent_system_template = self.cfg_template.agent_system_template
        self.agent_template = (
            getattr(self.cfg_template, "agent_template", None)
            or self.env_config.get("agent_template", "{observation}")
        )

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def run(self, seed: int, ppo_step: int, data: Dict[str, Any]) -> Dict[str, List[Any]]:
        """Run a single trajectory and return a rollout_batch dict."""
        rollout_cache = self._reset(seed, ppo_step, data)
        assert rollout_cache is not None

        while not rollout_cache.terminated:
            lm_output = self._make_decision(rollout_cache)
            stop_reason = lm_output.get("stop_reason")
            if stop_reason == GenerateStopReason.FINISH:
                rollout_cache = self._step(lm_output)
            if stop_reason == GenerateStopReason.MAX_LENGTH or stop_reason == GenerateStopReason.ABORT:
                break

        return self._formulate_rollout_batch(rollout_cache)

    # ------------------------------------------------------------------ #
    #  Internal methods (ported from trajectory_env_manager.TrajEnvManager)
    # ------------------------------------------------------------------ #

    def _build_env_and_llm_proxy(self) -> None:
        """Create ``env`` and ``llm_proxy`` for this trajectory (fresh each ``run``)."""
        if "seed" in self.env_config.get("config", {}):
            self.env_config["config"] = dict(self.env_config["config"])
            self.env_config["config"]["seed"] = self.env_config.get("group_seed", 0)
        self.env = gem.make(
            env_id=self.env_config["env_type"],
            **self.env_config.get("config", {}),
        )
        env_tool_config = self.rl_config.training.agentic.env_cfg_template.env_tool_config
        if env_tool_config is not None and env_tool_config.use_tools:
            self.env = ToolEnvWrapper(self.env, env_tool_config)

        agentic_config = self.rl_config.training.agentic
        self.llm_proxy = create_llm_proxy(
            sampler_client=self.sampler_client,
            llm_proxy_config=agentic_config.train_env_manager.llm_proxy,
            tokenizer=self.tokenizer,
            env=self.env,
        )

    def _reset(self, seed: int, ppo_step: int, data: Dict[str, Any]) -> RolloutCache:
        self._build_env_and_llm_proxy()

        self.rollout_cache = RolloutCache(
            env_id=self.env_config["env_id"],
            group_id=self.env_config["group_id"],
            tag=self.env_config["tag"],
        )

        observation, info, terminated = self.env.reset(seed=seed, ppo_step=ppo_step, data=data)
        assert observation is not None, "env.reset returned None observation"

        max_steps = self.env_config.get("max_steps", 20)
        self.rollout_cache.history.append({
            "observation": observation,
            "actions_left": max_steps - self.rollout_cache.step,
            "messages": [],
            **info,
        })
        if terminated:
            self.rollout_cache.terminated = True
            self.rollout_cache.terminated_reason = info.get("terminated_reason", None)
        return self.rollout_cache

    def _make_decision(self, rollout_cache: RolloutCache) -> Dict:
        input_ids = self._format_messages(rollout_cache)
        seq_length = self.rl_config.training.seq_length

        if input_ids.shape[1] >= seq_length:
            self.logger.warning(
                f"sequence_length = {seq_length} input_ids length = {input_ids.shape[1]}, "
                "maybe you should increase the response_length"
            )
            rollout_cache.terminated = True
            rollout_cache.terminated_reason = "EXCEED_MAX_LENGTH"
            return {"stop_reason": GenerateStopReason.MAX_LENGTH}

        prompt_token_ids = input_ids[0]
        lm_output = self.llm_proxy.generate({"prompt_token_ids": prompt_token_ids})

        if lm_output is None:
            log(f"[TrajEnvManager] _make_decision: lm_output is None")
            rollout_cache.terminated = True
            rollout_cache.terminated_reason = "LM_OUTPUT_IS_NONE"
            return {"stop_reason": GenerateStopReason.ABORT}

        response_ids = lm_output.get("response_ids")
        if response_ids is None:
            log(f"[TrajEnvManager] _make_decision: response_ids is None, {lm_output=}")
            rollout_cache.terminated = True
            rollout_cache.terminated_reason = "RESPONSE_IDS_IS_NONE"
            return {"stop_reason": GenerateStopReason.ABORT}
        if hasattr(response_ids, "tolist"):
            response_ids = response_ids.tolist()

        content = rollout_cache.history[-1]

        output_logprobs = lm_output.get("output_logprobs")
        if output_logprobs is not None:
            content["rollout_log_probs"] = (
                output_logprobs.tolist()
                if hasattr(output_logprobs, "tolist") else list(output_logprobs)
            )

        content["response_ids"] = response_ids
        content["messages"].append({
            "role": "assistant",
            "content": self.tokenizer.decode(response_ids, skip_special_tokens=True),
        })

        lm_output["stop_reason"] = GenerateStopReason.FINISH
        return lm_output

    def _step(self, lm_output: Dict) -> RolloutCache:
        response_ids = lm_output.get("response_ids")
        if hasattr(response_ids, "tolist"):
            response_ids = response_ids.tolist()
        responses = self.tokenizer.decode(
            response_ids if isinstance(response_ids, list) else response_ids,
            skip_special_tokens=False,
        )
        if not isinstance(responses, str):
            responses = responses[0] if responses else ""

        observation, reward, terminated, truncated, info = self.env.step(action=responses)
        suffix = info.pop("suffix", None)
        max_steps = self.env_config.get("max_steps", 20)

        self.rollout_cache.step += 1
        self.rollout_cache.terminated = terminated
        if terminated:
            self.rollout_cache.terminated_reason = info.get("terminated_reason", None)
        self.rollout_cache.truncated = truncated
        if self.rollout_cache.step >= max_steps:
            self.rollout_cache.terminated = True
            if not terminated:
                self.rollout_cache.terminated_reason = "EXCEED_MAX_STEPS"
                self.rollout_cache.truncated = True

        self.rollout_cache.history[-1]["reward"] = reward
        self.rollout_cache.history[-1]["llm_response"] = responses
        if info:
            self.rollout_cache.history[-1].update(info)

        self.rollout_cache.history.append({
            "observation": observation,
            "actions_left": max_steps - self.rollout_cache.step,
            "messages": [],
            "valid": True,
        })
        if suffix is not None:
            self.rollout_cache.history[-1]["suffix"] = suffix

        return self.rollout_cache

    def _format_messages(self, history: RolloutCache) -> torch.Tensor:
        """Build input_ids tensor from rollout history.

        Returns
        -------
        torch.Tensor
            ``(1, seq_len)`` int64 tensor of token IDs.
        """
        content = history.history[-1]
        max_steps = self.env_config.get("max_steps", 20)

        messages: List[Dict] = []
        user_content = ""
        if content["actions_left"] == max_steps:
            messages.append({"role": "system", "content": self.agent_system_template})
            if history.history and "env_instruction" in history.history[0]:
                user_content = f"{history.history[0]['env_instruction']}\n"
        if (
            len(history.history) > 1
            and history.history[-2].get("use_tool", False)
        ):
            messages.append({"role": "tool", "content": content["observation"]})
        else:
            render_dict: Dict[str, Any] = {"observation": content["observation"]}
            if contains_renderable_field(self.agent_template, "turn_idx"):
                render_dict["turn_idx"] = history.step + 1
            if contains_renderable_field(self.agent_template, "suffix"):
                render_dict["suffix"] = content.get("suffix", "")
            if contains_renderable_field(self.agent_template, "actions_left"):
                render_dict["actions_left"] = content["actions_left"]
            if contains_renderable_field(self.agent_template, "max_response_length"):
                render_dict["max_response_length"] = self.env_config.get(
                    "max_tokens_per_step", 512
                )
            user_content += self.agent_template.format(**render_dict)
            messages.append({"role": "user", "content": user_content})

        prompt_ids = custom_apply_chat_template(
            messages=messages,
            tokenizer=self.tokenizer,
            add_generation_prompt=True,
            tools=content.get("tools", None),
        )
        history_token_ids: List[int] = []
        for items in history.history[:-1]:
            history_token_ids.extend(items["prompt_ids"])
            history_token_ids.extend(items["response_ids"])
        # NOTE: 这里 response_ids 已经包含了 <|im_end|> token，所以不需要再加 <|im_end|> token
        # if len(history_token_ids):
        #     prompt_ids = compute_conversation_end_token_id(self.tokenizer) + prompt_ids
        input_ids = history_token_ids + prompt_ids

        content["prompt_ids"] = prompt_ids
        content["messages"] = messages

        return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

    # ------------------------------------------------------------------ #
    #  Output formatting
    # ------------------------------------------------------------------ #

    def _formulate_rollout_batch(
        self, rollout_cache: RolloutCache
    ) -> Dict[str, List[Any]]:
        """Pack one trajectory into a rollout_batch dict.

        Each value is a length-1 list (single sample).  The caller
        (``EnvAgentLoopActor``) merges multiple such dicts into a
        batched rollout_batch.
        """
        # Drop the trailing observation-only entry appended by _step()
        if (
            rollout_cache.history
            and "response_ids" not in rollout_cache.history[-1]
        ):
            rollout_cache.history.pop(-1)

        all_messages = [item["messages"] for item in rollout_cache.history]
        log(f"[TrajEnvManager] _formulate_rollout_batch: {all_messages}", rank=0)
        
        if len(rollout_cache.history) == 0:
            dummy_result = {
                "tokens": [torch.tensor([0], dtype=torch.long)],
                "prompt_lengths": [torch.tensor(0, dtype=torch.long)],
                "sequence_lengths": [torch.tensor(0, dtype=torch.long)],
                "rewards": [torch.tensor([0], dtype=torch.float)],
                "mask": [torch.tensor([0], dtype=torch.bool)],
                "rollout_log_probs": [torch.tensor([0.0], dtype=torch.float)],
            }
            return dummy_result

        scores = [item["reward"] for item in rollout_cache.history]

        token_ids: List[int] = []
        response_masks: List[int] = []
        rollout_log_probs: List[float] = []
        for item in rollout_cache.history:
            token_ids.extend(item["prompt_ids"])
            token_ids.extend(item["response_ids"])
            response_masks.extend(
                [0] * len(item["prompt_ids"]) + [1] * len(item["response_ids"])
            )
            if "rollout_log_probs" in item:
                rollout_log_probs.extend(
                    [0.0] * len(item["prompt_ids"]) + item["rollout_log_probs"]
                )

        tokens = torch.tensor(token_ids, dtype=torch.long)
        seq_len = len(token_ids)

        first_resp = response_masks.index(1)
        prompt_lengths = torch.tensor(first_resp, dtype=torch.long)
        sequence_lengths = torch.tensor(seq_len, dtype=torch.long)

        rewards = torch.tensor(sum(scores), dtype=torch.float)

        resp_mask = torch.tensor(response_masks, dtype=torch.bool)
        mask = resp_mask[1:]  # shifted by 1 to align with logprobs (L-1)

        result = {
            "tokens": [tokens],
            "prompt_lengths": [prompt_lengths],
            "sequence_lengths": [sequence_lengths],
            "rewards": [rewards],
            "mask": [mask],
        }

        if rollout_log_probs:
            rlp = torch.tensor(rollout_log_probs, dtype=torch.float)
            result["rollout_log_probs"] = [rlp[1:]]  # shift by 1 to align with logprobs

        return result
