import asyncio
import traceback
from contextlib import nullcontext
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer

from gpatch_v4.client import GenRmClient, SamplerClient
from gpatch_v4.configs.config import RlConfig
from gpatch_v4.rollout_generator.async_rollout.agent_loop_actor import (
    BaseAgentLoopActor,
)
from gpatch_v4.utils import GenerationAborted, import_fn_from_path, log

_DEFAULT_REFLECT_PROMPT = "请重新审视你的回答，仔细检查一下你的推理过程和最终答案。"


class TwoTurnReflectAgentLoopActor(BaseAgentLoopActor):
    """Two-turn rollout actor with reflection on best/worst samples.

    Pipeline per microbatch (``rollout_mbs == 1``):

    1. **First turn** — standard generation + gen_rm reward scoring.
    2. **Select** — pick top-k best and bottom-k worst samples by reward.
    3. **Build** — for each selected sample, append a reflection user
       message to form a new multi-turn prompt (token-level concat).
    4. **Second turn** — generate ``repeat_n`` responses for each of the
       ``2k`` new prompts, then score with gen_rm.
    5. **Merge** — return ``repeat_n + 2k * repeat_n`` samples total.
    """
    def __init__(self, config: RlConfig, worker_id: int):
        super().__init__(config, worker_id)
        self.sampler_client = None
        self.gen_rm_client = None
        self.tokenizer = None
        self.external_reward = None

    async def setup(self):
        self.sampler_client = SamplerClient(
            self.config, dp_rank=0, dp_size=1, skip_init_ipc_meta=True
        )
        if self.training_config.use_gen_rm_reward:
            self.gen_rm_client = GenRmClient(self.config, dp_rank=0, dp_size=1)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.policy.hf_tokenizer_path,
            use_fast=getattr(self.training_config, "use_fast_tokenizer", True),
            trust_remote_code=True,
        )

        reflect_prompt_text = getattr(
            self.training_config, "reflect_prompt", _DEFAULT_REFLECT_PROMPT
        )

        # Pre-compute the token sequence appended after the first turn's
        # <|im_end|>:  \n<|im_start|>user\n{reflect}<|im_end|>\n<|im_start|>assistant\n<think>...
        #
        # IMPORTANT: the sentinel uses only user messages (no assistant)
        # to avoid Qwen3 think-tag asymmetry — the template adds <think>
        # to the *last* assistant message but not to intermediate ones, so
        # a sentinel with an assistant message would cause prefix mismatch.
        _sentinel_user = [{"role": "user", "content": "x"}]
        _tmpl_kwargs = dict(tokenize=True, return_dict=False)
        _full = self.tokenizer.apply_chat_template(
            _sentinel_user + [{
                "role": "user",
                "content": reflect_prompt_text
            }],
            add_generation_prompt=True,
            **_tmpl_kwargs,
        )
        _prefix = self.tokenizer.apply_chat_template(
            _sentinel_user,
            add_generation_prompt=False,
            **_tmpl_kwargs,
        )
        assert _full[:len(_prefix)] == _prefix, (
            "Sentinel prefix mismatch — template may merge consecutive user turns"
        )
        # _prefix ends with ...<|im_end|>\n  — trim the \n so the
        # continuation carries it, bridging with first_turn's <|im_end|>.
        self.user_reflect_continuation: List[int] = _full[len(_prefix) - 1:]

        # Stop-token IDs used to normalise the tail of first-turn tokens.
        self.im_end_id: int = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        # 处理一些特殊情况
        _eot = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.strip_tail_ids = {self.im_end_id, _eot} - {None}

        self.reflect_top_k = self.training_config.reflect_top_k
        assert self.reflect_top_k is not None, f"{self.reflect_top_k=} must be set"

        # External reward (optional, runs in-process without distributed ctx)
        if self.training_config.use_external_reward:
            self._setup_external_reward()

        log(
            f"[TwoTurnReflectActor-{self.worker_id}] setup done, "
            f"reflect_top_k={self.reflect_top_k}, "
            f"external_reward={type(self.external_reward).__name__ if self.external_reward else None}, "
            f"reflect_continuation_len={len(self.user_reflect_continuation)}, "
            f"reflect_continuation_decoded="
            f"{self.tokenizer.decode(self.user_reflect_continuation, skip_special_tokens=False)!r}",
            rank=0
        )

    def _setup_external_reward(self):
        """Instantiate the external reward class from config."""
        external_reward_config = self.config.external_reward
        assert external_reward_config.reward_info is not None, (
            "external_reward.reward_info must be set when use_external_reward=True"
        )
        assert len(external_reward_config.reward_info
                  ) == 1, ("Currently only one external reward is supported")
        rm_info = external_reward_config.reward_info[0]
        assert rm_info.reward_py_path is not None, "reward_py_path must be set"
        assert rm_info.reward_cls_name is not None, "reward_cls_name must be set"

        reward_cls = import_fn_from_path(rm_info.reward_py_path, rm_info.reward_cls_name)
        assert hasattr(reward_cls, 'calc_external_reward'
                      ), (f"{rm_info.reward_cls_name} must implement calc_external_reward")
        self.external_reward = reward_cls(config=self.config, tokenizer=self.tokenizer)
        log(
            f"[TwoTurnReflectActor-{self.worker_id}] external reward initialized: {rm_info.reward_cls_name}"
        )

    # -------------------------------------------------------------- #
    #  Pipeline helpers (same pattern as AgentLoopActor)
    # -------------------------------------------------------------- #

    async def generate_batches(
        self,
        cleaned_batches: List[Dict[str, Any]],
        ppo_step: int,
        sample_indices: List[int],
        use_colocate: bool,
    ) -> List[Dict[str, List[Any]]]:
        """Generate rollouts by firing all requests, then awaiting all results."""
        sampler_idx = 0
        repeat_n = self.training_config.sampling_repeat_n
        if not cleaned_batches:
            return []
        ctx = self.sampler_phase(ppo_step) if use_colocate else nullcontext()
        async with ctx:
            refs = await asyncio.gather(
                *[
                    self.sampler_client.fire_generate(
                        sampler_idx,
                        ppo_step,
                        sidx,
                        cd,
                        repeat_n,
                        load_aware=self.training_config.load_aware_sampler_routing,
                    ) for cd, sidx in zip(cleaned_batches, sample_indices)
                ]
            )
            rbs = list(
                await asyncio.gather(*[self.sampler_client.await_generate(ref) for ref in refs])
            )
        return rbs

    async def _score_one_gen_rm(
        self,
        rbs: List[Dict[str, List[Any]]],
        ppo_step: int,
        sample_indices: List[int],
        rm_idx: int,
    ) -> List[Dict[str, List[Any]]]:
        """Score one gen-RM for all rollout batches."""
        if not rbs:
            return []
        return list(
            await asyncio.gather(
                *[
                    self.gen_rm_client.generate_rewards(rm_idx, ppo_step, sidx, rb)
                    for sidx, rb in zip(sample_indices, rbs)
                ]
            )
        )

    async def score_gen_rm_batches(
        self,
        rbs: List[Dict[str, List[Any]]],
        ppo_step: int,
        sample_indices: List[int],
        use_colocate: bool,
    ) -> List[Dict[str, List[Any]]]:
        """Apply gen-RM rewards, optionally wrapped in colocate lifecycle phases."""
        if not self.training_config.use_gen_rm_reward:
            return rbs
        phase_ctx = self.gen_rm_all_phase(ppo_step) if use_colocate else nullcontext()
        async with phase_ctx:
            all_results = await asyncio.gather(
                *[
                    self._score_one_gen_rm(rbs, ppo_step, sample_indices, rm_idx)
                    for rm_idx in range(self.gen_rm_client.num_rms)
                ]
            )
        for rm_results in all_results:
            for rb, result in zip(rbs, rm_results):
                rb.update(result)
        return rbs

    async def score_external_reward_batches(
        self,
        rbs: List[Dict[str, List[Any]]],
        ppo_step: int,
    ) -> List[Dict[str, List[Any]]]:
        """Apply CPU external rewards for all rollout batches."""
        if self.external_reward is None:
            return rbs
        all_updates = await asyncio.gather(
            *[self.external_reward.calc_external_reward([rb], ppo_step) for rb in rbs]
        )
        for rb, updates in zip(rbs, all_updates):
            assert len(updates) == 1, (
                f"calc_external_reward must return one update per input batch; got {len(updates)}"
            )
            rb.update(updates[0])
        return rbs

    async def score_rollout_batches(
        self,
        rbs: List[Dict[str, List[Any]]],
        ppo_step: int,
        sample_indices: List[int],
        use_colocate: bool,
    ) -> List[Dict[str, List[Any]]]:
        """Shared reward pipeline: gen_rm + external reward."""
        rbs = await self.score_gen_rm_batches(
            rbs,
            ppo_step,
            sample_indices,
            use_colocate=use_colocate,
        )
        return await self.score_external_reward_batches(rbs, ppo_step)

    # -------------------------------------------------------------- #
    #  Main entry point
    # -------------------------------------------------------------- #

    async def agent_loop(
        self,
        cleaned_data: Dict[str, Any] | List[Dict[str, Any]],
        ppo_step: int,
        microbatch_idx: int | List[int],
        sample_idx: int | List[int],
        use_colocate: bool = False,
    ) -> List[Dict[str, List[Any]]]:
        """Two-turn entry for both disaggregated and colocate modes.

        Disaggregated callers pass one microbatch. Colocate callers pass
        this agent's list of microbatches and set ``use_colocate=True``.
        """
        try:
            if isinstance(cleaned_data, list):
                cleaned_batches = cleaned_data
                sample_indices = sample_idx
            else:
                cleaned_batches = [cleaned_data]
                sample_indices = [sample_idx]
            assert isinstance(sample_indices,
                              list), ("sample_idx must be a list when cleaned_data is a list")
            return await self._batch_loop(
                cleaned_batches,
                ppo_step,
                sample_indices,
                use_colocate=use_colocate,
            )
        except Exception as e:
            traceback.print_exc()
            raise e

    async def _batch_loop(
        self,
        cleaned_batches: List[Dict[str, Any]],
        ppo_step: int,
        sample_indices: List[int],
        use_colocate: bool,
    ) -> List[Dict[str, List[Any]]]:
        """Two-turn pipeline unified for both disaggregated and colocate.

        Turn 1: [sampler] → [gen_rm] → [external_reward] → CPU select
        Turn 2: [sampler] → [gen_rm] → [external_reward]
        Merge:  return turn1 + turn2 rbs interleaved per microbatch.
        """
        k = self.reflect_top_k

        # ---- Turn 1: generate ----
        rbs_t1 = await self.generate_batches(
            cleaned_batches,
            ppo_step,
            sample_indices,
            use_colocate,
        )
        if self.worker_id == 0:
            log(
                f"[TwoTurnReflect-{self.worker_id}] turn1 sampler done: "
                f"{ppo_step=}, {len(cleaned_batches)} mbs, "
                f"mode={'colocate' if use_colocate else 'disaggregated'}"
            )

        # ---- Turn 1: score ----
        rbs_t1 = await self.score_rollout_batches(
            rbs_t1,
            ppo_step,
            sample_indices,
            use_colocate,
        )

        # Turn 2 issues brand-new sampler requests, which a hard abort can no
        # longer reach, so the cooperative flag is the only way to stop a
        # microbatch that survived into the gap between the two turns.
        if self._pause_event.is_set():
            raise GenerationAborted(
                f"[TwoTurnReflect-{self.worker_id}] paused before turn 2, "
                f"dropping {len(cleaned_batches)} microbatch(es) at {ppo_step=}"
            )

        # ---- Selection + build reflect prompts (CPU) ----
        reward_key = "rewards"
        all_reflect_cleaned: List[Dict[str, Any]] = []
        all_reflect_indices: List[int] = []
        selected_per_mb: List[list] = []

        for rbi, rb in enumerate(rbs_t1):
            best, worst = self._select_topk_bottomk(rb[reward_key], k)
            selected = ([(idx, "best") for idx in best] + [(idx, "worst") for idx in worst])
            selected_per_mb.append(selected)
            for offset, (sel_idx, tag) in enumerate(selected):
                new_cleaned = self._build_reflect_prompt(rb, sel_idx, tag)
                t2_sidx = sample_indices[rbi] * 10000 + offset + 1
                all_reflect_cleaned.append(new_cleaned)
                all_reflect_indices.append(t2_sidx)

        if self.worker_id == 0:
            log(
                f"[TwoTurnReflect-{self.worker_id}] selection done: "
                f"{len(all_reflect_cleaned)} reflect prompts"
            )

        # ---- Turn 2: generate ----
        rbs_t2 = await self.generate_batches(
            all_reflect_cleaned,
            ppo_step,
            all_reflect_indices,
            use_colocate,
        )
        if self.worker_id == 0:
            log(f"[TwoTurnReflect-{self.worker_id}] turn2 sampler done: "
                f"{len(rbs_t2)} rbs")

        # ---- Turn 2: score ----
        rbs_t2 = await self.score_rollout_batches(
            rbs_t2,
            ppo_step,
            all_reflect_indices,
            use_colocate,
        )

        # ---- Tag parent unique_id on turn2 rbs ----
        t2_offset = 0
        for rbi, (rb_t1, selected) in enumerate(zip(rbs_t1, selected_per_mb)):
            for sel_idx, _tag in selected:
                rb_t2 = rbs_t2[t2_offset]
                parent_uid = rb_t1["unique_id"][sel_idx]
                rb_t2["parent_unique_id"] = [parent_uid] * len(rb_t2["tokens"])
                t2_offset += 1

        # ---- Merge: interleave turn1 rb + its turn2 rbs ----
        all_rbs: List[Dict[str, List[Any]]] = []
        t2_offset = 0
        for rbi, (rb_t1, selected) in enumerate(zip(rbs_t1, selected_per_mb)):
            all_rbs.append(rb_t1)
            for _ in selected:
                all_rbs.append(rbs_t2[t2_offset])
                t2_offset += 1

        total = sum(len(rb["tokens"]) for rb in all_rbs)
        if self.worker_id == 0:
            log(
                f"[TwoTurnReflect-{self.worker_id}] done: "
                f"{len(all_rbs)} batches, total_samples={total}"
            )
        return all_rbs

    # -------------------------------------------------------------- #
    #  Helpers
    # -------------------------------------------------------------- #

    def _select_topk_bottomk(self, rewards: List[Any], k: int) -> tuple:
        """Return indices of top-k highest and bottom-k lowest rewards."""
        reward_vals = torch.tensor(
            [r.item() if isinstance(r, torch.Tensor) else float(r) for r in rewards]
        )
        n = reward_vals.numel()
        actual_k = min(k, n)
        best_indices = torch.topk(reward_vals, actual_k).indices.tolist()
        worst_indices = torch.topk(reward_vals, actual_k, largest=False).indices.tolist()
        return best_indices, worst_indices

    def _build_reflect_prompt(
        self,
        rb_turn1: Dict[str, List[Any]],
        sample_idx: int,
        tag: str,
    ) -> Dict[str, Any]:
        """Build a new cleaned_data dict for a second-turn reflection prompt.

        Uses raw first-turn tokens (preserving ``<think>`` tags) and
        appends the pre-computed ``user_reflect_continuation``.

        Result layout::

            tokens[:seq_len]  +  user_reflect_continuation
            ───────────────      ──────────────────────────
            sys + user(q)        \\n
            + gen_prompt         <|im_start|>user
            + response           {reflect_prompt}<|im_end|>\\n
            + <|im_end|>         <|im_start|>assistant\\n<think>...
        """
        tokens_tensor = rb_turn1["tokens"][sample_idx]
        seq_len = rb_turn1["sequence_lengths"][sample_idx]
        if isinstance(seq_len, torch.Tensor):
            seq_len = seq_len.item()

        first_turn_ids = tokens_tensor[:seq_len].tolist()

        # Normalise the tail: strip <|endoftext|>, ensure ends with <|im_end|>.
        while first_turn_ids and first_turn_ids[-1] in self.strip_tail_ids:
            first_turn_ids.pop()
        first_turn_ids.append(self.im_end_id)

        new_prompt_ids = first_turn_ids + self.user_reflect_continuation
        original_uid = rb_turn1["unique_id"][sample_idx]

        if self.worker_id == 0:
            log(
                f"[TwoTurnReflectActor-{self.worker_id}] "
                f"reflect prompt ({tag}, idx={sample_idx}, uid={original_uid}):\n"
                f"two_turns_prompt {self.tokenizer.decode(new_prompt_ids, skip_special_tokens=False)}",
                rank=0,
            )

        return {
            "prompt_token_ids": [{
                "prompt_token_ids": new_prompt_ids
            }],
            "prompt_lens": [torch.tensor(len(new_prompt_ids), dtype=torch.long)],
            "gt_label": [rb_turn1["gt_label"][sample_idx]],
            "unique_id": [f"{original_uid}_reflect_{tag}_{sample_idx}"],
        }
