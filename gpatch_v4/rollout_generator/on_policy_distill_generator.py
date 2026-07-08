import asyncio
import os
from typing import Any, Dict, List

import torch
import torch.distributed
from typing_extensions import override

from gpatch_v4.configs.config import OnPolicyDistillConfig
from gpatch_v4.core.parallel_state import cpu_barrier, is_mp_and_cp_head
from gpatch_v4.rollout_generator.base_generator import BaseRolloutGenerator
from gpatch_v4.utils import (
    TimerSingleton,
    clear_memory,
    destroy_process_groups,
    logging_memory_usage,
    logging_memory_usage_details,
    reload_process_groups,
)
from gpatch_v4.utils.training_utils import check_rollout_batches


class OnPolicyDistillRolloutGenerator(BaseRolloutGenerator):
    """Rollout generator for on-policy distillation with multi-teacher support.

    Each teacher's log probs are stored in the rollout batch as ``teacher_logprobs_{name}``.

    Parameters
    ----------
    config : OnPolicyDistillConfig
    sampler_client : object
    gen_rm_client : object
    bt_rm_client : object
    run_eval : bool, optional
    teacher_clients : dict[str, object]
    """
    def __init__(
        self,
        config: OnPolicyDistillConfig,
        sampler_client,
        gen_rm_client,
        bt_rm_client,
        run_eval=False,
        teacher_clients=None,
    ):
        super().__init__(config, sampler_client, gen_rm_client, bt_rm_client, run_eval)
        assert teacher_clients and len(teacher_clients) > 0, "At least one teacher is required"
        self.teacher_clients: Dict[str, Any] = teacher_clients

    async def calc_all_teacher_logps(
        self, rbs, num_microbatches, curr_ppo_step, sample_idx_base: int = None
    ):
        """Compute log probs from all teachers in parallel.

        Fans out requests to every teacher concurrently per phase, with a
        single ``cpu_barrier`` between phases for deterministic ordering.

        ``sample_idx_base`` overrides ``self.sample_idx`` when log_prob_top_k>0.
        当log_prob_top_k>0时， compute_teacher_logps需要student logits topk-ids, 
        所以需要在 __call__ 之外，compute_student_logps之后再调用这个函数，但这时，
        self.sample_idx 已经被加过了，不能直接用，所以需要传一个 sample_idx_base 来覆盖它.
        """
        teacher_items = list(self.teacher_clients.items())

        # Phase 1: wake up all teachers concurrently
        await asyncio.gather(
            *[t_client.mark_ppo_step_begin(0, curr_ppo_step) for _, t_client in teacher_items]
        )
        cpu_barrier()
        logging_memory_usage_details(
            f"memory tracking actor after teacher wake_up",
            rank=0,
        )

        # Phase 2: issue calc_logps requests for all teachers concurrently
        # Only send keys that teacher actually needs to reduce RPC transfer size.
        if self.is_mp_and_cp_head:
            s_idx = self.sample_idx if sample_idx_base is None else sample_idx_base
            teacher_keys = {
                "tokens", "sequence_lengths", "stu_topk_ids",
                "position_ids", "image_input_mask",
                "vision_data", "vision_grid_thw",
                "input_features", "feature_attention_mask",
                "audio_feature",
            }
            all_issue_cos = []
            for _, t_client in teacher_items:
                for rbi, rollout_batch in enumerate(rbs):
                    lightweight_batch = {
                        k: rollout_batch[k] for k in teacher_keys if k in rollout_batch
                    }
                    # Truncate stu_topk_ids to response length for transfer efficiency.
                    if "stu_topk_ids" in lightweight_batch:
                        seq_lens = lightweight_batch["sequence_lengths"]
                        lightweight_batch["stu_topk_ids"] = [
                            ids[:int(sl.item()) - 1]
                            for ids, sl in zip(lightweight_batch["stu_topk_ids"], seq_lens)
                        ]
                    all_issue_cos.append(
                        t_client.issue_calc_logps(
                            lightweight_batch,
                            curr_ppo_step,
                            s_idx + rbi,
                        )
                    )
            await asyncio.gather(*all_issue_cos)
        cpu_barrier()

        # Phase 3: collect results for all teachers concurrently, then
        # update rollout_batches in deterministic order
        if self.is_mp_and_cp_head:
            s_idx = self.sample_idx if sample_idx_base is None else sample_idx_base
            all_result_cos = []
            for _, t_client in teacher_items:
                for rbi in range(num_microbatches):
                    all_result_cos.append(
                        t_client.get_calc_logps_result(curr_ppo_step, s_idx + rbi)
                    )
            all_results = await asyncio.gather(*all_result_cos)

            idx = 0
            for t_name, _ in teacher_items:
                for rbi, rollout_batch in enumerate(rbs):
                    resp = all_results[idx]
                    rollout_batch[f"teacher_logprobs_{t_name}"] = resp["teacher_logprobs"]
                    if "teacher_on_stu_topk_logprobs" in resp:
                        rollout_batch[f"teacher_on_stu_topk_logprobs_{t_name}"] = (
                            resp["teacher_on_stu_topk_logprobs"]
                        )
                    if "teacher_topk_ids" in resp:
                        rollout_batch[f"teacher_topk_ids_{t_name}"] = resp["teacher_topk_ids"]
                    if "teacher_topk_logprobs" in resp:
                        rollout_batch[f"teacher_topk_logprobs_{t_name}"] = (
                            resp["teacher_topk_logprobs"]
                        )
                    idx += 1
                assert check_rollout_batches(rbs), \
                    f"rbs format error after teacher {t_name}"
        cpu_barrier()

        # Phase 4: put all teachers to sleep concurrently
        await asyncio.gather(
            *[t_client.mark_ppo_step_end(0, curr_ppo_step) for _, t_client in teacher_items]
        )
        cpu_barrier()

        logging_memory_usage_details("memory tracking actor after teacher sleep", rank=0)

        return rbs

    @override
    def _post_process_rm_rollout_batch(self, rollout_batches: List[Dict[str, List[Any]]]):
        return self.apply_sampling_rollout_attr.add_back_rollout_attr(rollout_batches)

    @override
    async def __call__(self, data_iter, num_microbatches, curr_ppo_step):
        timers = TimerSingleton.get_timer()
        offload_process_group = self.config.training.offload_process_group
        if offload_process_group:
            destroy_process_groups()
            clear_memory()

        timers("sampler_generate", log_level=0).start(barrier=True)
        rbs = await self.rollout_samples(data_iter, num_microbatches, curr_ppo_step)
        if self.is_mp_and_cp_head:
            rbs = self._hook_after_sampling(rbs, curr_ppo_step)
            assert check_rollout_batches(rbs), f"rbs format error, may need pop('ready'): {rbs=}"
        cpu_barrier()
        timers("sampler_generate").stop()

        timers("gen_rm_generate", log_level=0).start(barrier=True)
        if self.training_config.use_gen_rm_reward:
            rbs = await self.generate_gen_rm_reward(rbs, num_microbatches, curr_ppo_step)
        cpu_barrier()
        timers("gen_rm_generate").stop()

        timers("bt_rm_generate", log_level=0).start(barrier=True)
        if self.training_config.use_bt_rm_reward:
            rbs = await self.calc_bt_rm_reward(rbs, num_microbatches, curr_ppo_step)
        cpu_barrier()
        timers("bt_rm_generate").stop()

        if self.is_mp_and_cp_head:
            rbs = self._post_process_rm_rollout_batch(rbs)

        # Teacher runs here when it does NOT depend on student's topk_ids.
        # only_tch: teacher produces its own topk (no dependency on student).
        # only_stu/intersection/union: teacher needs stu_topk_ids → deferred to actor.
        if self.config.ppo.log_prob_top_k == 0 or self.config.ppo.opd_top_k_strategy == "only_tch":
            timers("compute_teacher_logps", log_level=0).start(barrier=True)
            rbs = await self.calc_all_teacher_logps(rbs, num_microbatches, curr_ppo_step)
            cpu_barrier()
            timers("compute_teacher_logps").stop()

        if self.is_mp_and_cp_head:
            for rb in rbs:
                self.apply_sampling_rollout_attr.remove_rollout_attr(rb)
        self.sample_idx += num_microbatches

        if offload_process_group:
            reload_process_groups()
        return rbs
