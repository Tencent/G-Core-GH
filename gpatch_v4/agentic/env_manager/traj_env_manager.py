"""Simplified TrajEnvManager for async rollout architecture.

Runs a single trajectory and returns a ``rollout_batch`` dict directly.
Designed to be driven by ``EnvAgentLoopActor`` (e.g. ``asyncio.to_thread``).
"""

from contextlib import suppress
from typing import Any, Dict, List, Optional

with suppress(ImportError):
    import gem

import torch
from transformers import PreTrainedTokenizer

from gpatch_v4.agentic.env.tool_reason_env_wrapper import ToolReasonEnvWrapper
from gpatch_v4.agentic.env_manager.token_mask_utils import (
    compute_conversation_end_token_id,
    custom_apply_chat_template,
)
from gpatch_v4.agentic.env_manager.utils import EnvManagerStrMixin, RolloutCache
from gpatch_v4.agentic.llm_proxy import create_llm_proxy
from gpatch_v4.client import SamplerClient
from gpatch_v4.configs.config import AgenticRlConfig
from gpatch_v4.utils import log, log_debug
from gpatch_v4.utils.constants import GenerateStopReason
from gpatch_v4.utils.data_manipulate_utils import aggregate_metrics
from gpatch_v4.utils.str_utils import contains_renderable_field


def make_invalid_traj_dummy_batch(
    metric_schema=(),
    pad_token_id=None,
    routed_experts_shape=None,
):
    """Build a THD/BSHD-safe placeholder for empty or failed trajectories.

    GRPO THD packing requires ``sequence_lengths == tokens.numel() >= 2``.
    The response mask is all-False so the sample contributes no PPO loss.
    A length-2 placeholder also gets a non-zero padded budget in
    ``convert_mbs_for_pack_seq`` (unlike the old ``sequence_lengths=0`` dummy).
    """
    pad_id = 0 if pad_token_id is None else int(pad_token_id)
    # 2 tokens → shifted axis length 1 for mask / rollout_log_probs.
    tokens = torch.tensor([pad_id, pad_id], dtype=torch.long)
    result = {
        "tokens": [tokens],
        "prompt_lengths": [torch.tensor(1, dtype=torch.long)],
        "sequence_lengths": [torch.tensor(2, dtype=torch.long)],
        "rewards": [torch.tensor(0.0, dtype=torch.float)],
        "mask": [torch.tensor([False], dtype=torch.bool)],
        "position_ids": [
            torch.arange(2, dtype=torch.long).view(1, 1, 2).expand(3, 1, 2).contiguous()
        ],
        "image_input_mask": [torch.zeros(1, 2, dtype=torch.bool)],
        "rollout_log_probs": [torch.tensor([0.0], dtype=torch.float)],
    }
    if routed_experts_shape is not None:
        num_layers, moe_router_topk = routed_experts_shape
        assert num_layers > 0 and moe_router_topk > 0, routed_experts_shape
        dummy_experts = torch.arange(moe_router_topk, dtype=torch.int32)
        result["routed_experts"] = [
            dummy_experts.view(1, 1, moe_router_topk).expand(
                2, num_layers, moe_router_topk
            ).contiguous()
        ]
    for name in metric_schema or ():
        result[name] = [torch.tensor(0.0, dtype=torch.float)]
    return result


class TrajEnvManager(EnvManagerStrMixin):
    """Lightweight env manager that runs exactly one trajectory.

    Parameters
    ----------
    rl_config : AgenticRlConfig
    env_config : dict
        Per-env configuration (env_type, config, env_id, group_id, tag,
        max_steps, ``engine_index``, etc.).  ``engine_index`` is assigned
        by ``EnvAgentLoopActor`` per trajectory dispatch and forwarded to
        ``EngineProxy.generate`` as the sglang engine routing hint.
    tokenizer : PreTrainedTokenizer
        Should be a dedicated copy per manager for thread safety.
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
        self.rollout_cache: Optional[RolloutCache] = None
        self.env = None
        self.llm_proxy = None

        agentic_config = rl_config.training.agentic
        self.cfg_template = agentic_config.env_cfg_template
        self.agent_system_template = self.cfg_template.agent_system_template
        self.agent_template = (
            getattr(self.cfg_template, "agent_template", None) or
            self.env_config.get("agent_template", "{observation}")
        )

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def _router_replay_shape(self) -> Optional[tuple[int, int]]:
        """Return sampler routing ``(num_layers, topk)`` when replay is enabled."""
        if not self.rl_config.training.moe_router_replay:
            return None

        num_layers = self.rl_config.training.moe_router_replay_num_layers
        moe_router_topk = self.rl_config.training.moe_router_replay_topk
        if num_layers is None or moe_router_topk is None:
            raise ValueError(
                "MoE router replay layout was not resolved before agentic rollout; "
                "RolloutController.setup must populate "
                "training.moe_router_replay_num_layers/topk"
            )
        return int(num_layers), int(moe_router_topk)

    def run(self, seed: int, ppo_step: int, data: Dict[str, Any]) -> Dict[str, List[Any]]:
        """Run a single trajectory and return a rollout_batch dict."""
        self.sampling_seed_offset = seed * 10 + self.env_config["env_id"]
        rollout_cache = self._reset(seed, ppo_step, data)
        assert rollout_cache is not None

        while not rollout_cache.terminated:
            lm_output = self._make_decision(rollout_cache)
            stop_reason = lm_output.get("stop_reason")
            if stop_reason in (
                GenerateStopReason.FINISH,
                GenerateStopReason.MAX_GEN_LENGTH,
            ):
                # Both reasons contain a sampled response. Let env.step decide
                # whether consuming that response terminates the trajectory.
                rollout_cache = self._step(lm_output)
            elif stop_reason in (
                GenerateStopReason.MAX_LENGTH,
                GenerateStopReason.ABORT,
            ):
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

        env_kwargs = dict(self.env_config.get("config", {}))
        # ``reasoning_parser`` lives inside env_cfg_template.env_config (a Dict)
        # so it would otherwise be unpacked as a kwarg to gem.make; intercept
        # it here so the inner env constructor doesn't see it.
        reasoning_parser_name = env_kwargs.pop("reasoning_parser", "") or ""
        self.env = gem.make(
            env_id=self.env_config["env_type"],
            **env_kwargs,
        )
        env_tool_config = self.rl_config.training.agentic.env_cfg_template.env_tool_config
        use_tools = env_tool_config is not None and env_tool_config.use_tools
        enable_thinking = bool(self.rl_config.training.enable_thinking)
        if use_tools or (enable_thinking and reasoning_parser_name):
            self.env = ToolReasonEnvWrapper(
                self.env,
                env_tool_config=env_tool_config if use_tools else None,
                reasoning_parser_name=reasoning_parser_name,
                enable_thinking=enable_thinking,
            )

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
        self.rollout_cache.history.append(
            {
                "observation": observation,
                "actions_left": max_steps - self.rollout_cache.step,
                "messages": [],
                **info,
            }
        )
        if terminated:
            self.rollout_cache.terminated = True
            self.rollout_cache.terminated_reason = info.get("terminated_reason", None)
        return self.rollout_cache

    def _make_decision(self, rollout_cache: RolloutCache) -> Dict:
        input_ids = self._format_messages(rollout_cache)
        seq_length = self.rl_config.training.seq_length

        if input_ids.shape[1] >= seq_length:
            log(
                f"sequence_length = {seq_length} input_ids length = {input_ids.shape[1]}, "
                "maybe you should increase the response_length"
            )
            rollout_cache.terminated = True
            rollout_cache.terminated_reason = "EXCEED_MAX_LENGTH"
            return {"stop_reason": GenerateStopReason.MAX_LENGTH}

        # 从源头 cap 住 response 长度，防止 trajectory 累计 tokens 超过 seq_length。
        # 注意：sampler 端的 generate_func 必须读这个 key 并用它去 override 引擎默认
        # 的 max_new_tokens。若 generate_func 未实现该逻辑，此处只是冗余信息，无害。
        prompt_token_ids = input_ids[0]
        max_tokens_per_step = self.env_config["max_tokens_per_step"]
        max_new_tokens_cap = seq_length - input_ids.shape[1]
        if max_tokens_per_step is not None and max_tokens_per_step > 0:
            max_new_tokens_cap = min(max_new_tokens_cap, max_tokens_per_step)
        # seed_offset 用 env id 来标志，让一个轨迹一个seed，不管多少轮，不同轨迹不同 seed
        lm_output = self.llm_proxy.generate(
            {
                "prompt_token_ids": prompt_token_ids,
                "max_new_tokens": max_new_tokens_cap,
                "seed_offset": self.sampling_seed_offset,
            },
            engine_index=self.env_config["engine_index"],
        )

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

        if self.rl_config.training.moe_router_replay:
            routed_experts = lm_output.pop("routed_experts", None)
            assert routed_experts is not None, (
                "moe_router_replay is enabled, but the agentic sampler response "
                "does not contain routed_experts"
            )
            assert torch.is_tensor(routed_experts) and routed_experts.ndim == 3, (
                "routed_experts must be a [seq, layer, topk] tensor, got "
                f"{type(routed_experts)=}, "
                f"shape={getattr(routed_experts, 'shape', None)}"
            )
            expected_seq_len = input_ids.shape[1] + len(response_ids)
            assert routed_experts.shape[0] == expected_seq_len, (
                "agentic routing must cover the complete sampler request: "
                f"{routed_experts.shape[0]=} != {expected_seq_len=}"
            )
            expected_shape = self._router_replay_shape()
            assert expected_shape is not None
            assert tuple(routed_experts.shape[1:]) == expected_shape, (
                "agentic routing layer/topk shape mismatch: "
                f"{tuple(routed_experts.shape[1:])=} != {expected_shape=}"
            )

            # The sampler returns routing for the complete request prefix plus
            # this turn's response. Keep only the current turn's suffix here so
            # every history item remains self-contained when a trajectory is
            # truncated or branched. clone() avoids retaining the full tensor's
            # backing storage through a view.
            current_turn_seq_len = len(content["prompt_ids"]) + len(response_ids)
            current_turn_start = expected_seq_len - current_turn_seq_len
            assert current_turn_start >= 0
            content["routed_experts"] = routed_experts[current_turn_start:].clone()

        output_logprobs = lm_output.get("output_logprobs")
        if output_logprobs is not None:
            content["rollout_log_probs"] = (
                output_logprobs.tolist()
                if hasattr(output_logprobs, "tolist") else list(output_logprobs)
            )

        content["response_ids"] = response_ids
        content["messages"].append(
            {
                "role": "assistant",
                "content": self.tokenizer.decode(response_ids, skip_special_tokens=True),
            }
        )
        log_debug(
            f"make decision input content: {self.tokenizer.decode(input_ids[0], skip_special_tokens=False)}"
        )
        log_debug(
            f"make decision output content: {self.tokenizer.decode(response_ids, skip_special_tokens=False)}"
        )
        engine_finish_reason = lm_output.get("engine_finish_reason")
        if engine_finish_reason == "length":
            lm_output["stop_reason"] = GenerateStopReason.MAX_GEN_LENGTH
        elif engine_finish_reason == "abort":
            lm_output["stop_reason"] = GenerateStopReason.ABORT
        else:
            # SGLang uses "stop" for EOS / a configured stop token. Keep the
            # historical fallback for backends that do not expose a reason.
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

        self.rollout_cache.history.append(
            {
                "observation": observation,
                "actions_left": max_steps - self.rollout_cache.step,
                "messages": [],
                "valid": True,
            }
        )
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
        if (len(history.history) > 1 and history.history[-2].get("use_tool", False)):
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
                render_dict["max_response_length"] = self.env_config["max_tokens_per_step"]
            user_content += self.agent_template.format(**render_dict)
            messages.append({"role": "user", "content": user_content})

        prompt_ids = custom_apply_chat_template(
            messages=messages,
            tokenizer=self.tokenizer,
            add_generation_prompt=True,
            tools=content.get("tools", None),
            enable_thinking=self.rl_config.training.enable_thinking,
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

    def _formulate_rollout_batch(self, rollout_cache: RolloutCache) -> Dict[str, List[Any]]:
        """Pack one trajectory into a rollout_batch dict.

        Each value is a length-1 list (single sample).  The caller
        (``EnvAgentLoopActor``) merges multiple such dicts into a
        batched rollout_batch.
        """
        # Drop the trailing observation-only entry appended by _step()
        if (rollout_cache.history and "response_ids" not in rollout_cache.history[-1]):
            rollout_cache.history.pop(-1)

        if len(rollout_cache.history) == 0:
            assert self.tokenizer is not None
            tokenizer = self.tokenizer
            pad_token_id = getattr(tokenizer, "pad_token_id", 0)
            metric_schema = getattr(self, "metric_schema", ())
            return make_invalid_traj_dummy_batch(
                metric_schema,
                pad_token_id,
                routed_experts_shape=self._router_replay_shape(),
            )

        scores = [item["reward"] for item in rollout_cache.history]

        token_ids: List[int] = []
        response_masks: List[int] = []
        rollout_log_probs: List[float] = []
        for item in rollout_cache.history:
            token_ids.extend(item["prompt_ids"])
            token_ids.extend(item["response_ids"])
            response_masks.extend([0] * len(item["prompt_ids"]) + [1] * len(item["response_ids"]))
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

        seq_length = self.rl_config.training.seq_length
        pad_multi = self.rl_config.training.pad_to_mulitiple_of

        L = tokens.shape[-1]
        pad_to_len = ((L + pad_multi - 1) // pad_multi) * pad_multi
        pos_len = max(seq_length, pad_to_len)
        position_ids = (
            torch.arange(pos_len, dtype=torch.long).view(1, 1, pos_len).expand(3, 1,
                                                                               pos_len).contiguous()
        )
        image_input_mask = torch.zeros(1, pos_len, dtype=torch.bool)

        result = {
            "tokens": [tokens],
            "prompt_lengths": [prompt_lengths],
            "sequence_lengths": [sequence_lengths],
            "rewards": [rewards],
            "mask": [mask],
            "position_ids": [position_ids],
            "image_input_mask": [image_input_mask],
        }

        if self.rl_config.training.moe_router_replay:
            expected_shape = self._router_replay_shape()
            assert expected_shape is not None
            routed_experts_by_turn = []
            for turn_idx, item in enumerate(rollout_cache.history):
                turn_routed_experts = item.get("routed_experts")
                assert turn_routed_experts is not None, (
                    "moe_router_replay is enabled, but trajectory turn "
                    f"{turn_idx} has no routed_experts"
                )
                assert torch.is_tensor(turn_routed_experts) and turn_routed_experts.ndim == 3
                turn_seq_len = len(item["prompt_ids"]) + len(item["response_ids"])
                assert turn_routed_experts.shape[0] == turn_seq_len, (
                    "agentic routing is not token-aligned for trajectory turn "
                    f"{turn_idx}: {turn_routed_experts.shape[0]=} != {turn_seq_len=}"
                )
                assert tuple(turn_routed_experts.shape[1:]) == expected_shape, (
                    "agentic routing layer/topk shape mismatch for trajectory turn "
                    f"{turn_idx}: {tuple(turn_routed_experts.shape[1:])=} != "
                    f"{expected_shape=}"
                )
                routed_experts_by_turn.append(turn_routed_experts)

            routed_experts = torch.cat(routed_experts_by_turn, dim=0)
            assert routed_experts.shape[0] == seq_len, (
                "final agentic routing is not token-aligned: "
                f"{routed_experts.shape[0]=} != {seq_len=}"
            )
            result["routed_experts"] = [routed_experts]

        if rollout_log_probs:
            rlp = torch.tensor(rollout_log_probs, dtype=torch.float)
            result["rollout_log_probs"] = [rlp[1:]]  # shift by 1 to align with logprobs

        # Aggregate per-step env metrics (e.g. success/action_is_valid/format_penalty)
        # into trajectory-level scalars and surface them on the rollout_batch so
        # that ``compute_rollout_metrics`` (which scans for keys listed in
        # ``training.metrics_report``) can all-reduce and report them.
        history_metrics: List[Dict[str, Any]] = []
        agg_mode: Dict[str, str] = {}
        for item in rollout_cache.history:
            m = item.get("metrics")
            if isinstance(m, dict) and m:
                history_metrics.append(m)
                item_mode = item.get("metrics_agg_mode")
                if isinstance(item_mode, dict):
                    agg_mode.update(item_mode)
        if history_metrics:
            traj_metrics = aggregate_metrics(history_metrics, agg_mode)
            for name, val in traj_metrics.items():
                result[name] = [torch.tensor(float(val), dtype=torch.float)]

        result["num_actions"] = [torch.tensor(float(rollout_cache.step), dtype=torch.float)]

        return result
