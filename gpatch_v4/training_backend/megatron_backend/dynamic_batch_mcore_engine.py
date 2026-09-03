from typing import Any, Dict, List

import torch

from megatron.core import mpu

from gpatch_v4.core.adaptive_entropy import update_adaptive_entropy_after_train_step
from gpatch_v4.core.smart_pad_helper import CatedSmartPadInferHelper
from gpatch_v4.utils import clear_memory, extend_value_to_dict
from gpatch_v4.utils.communication_utils import BroadcastUtils

from .dynamic_batch_mixin import DynamicBatchMcoreMixin
from .mcore_engine import McoreEngine, clear_gathered_routing_info


class DynamicBatchMcoreEngine(DynamicBatchMcoreMixin, McoreEngine):
    """MCore engine whose public RL interfaces consume dynamic batches."""
    def compute_log_probs(
        self,
        samples: List[Dict[str, Any]],
        compute_pre_logps: bool = True,
        policy_compute_topk: bool = False,
        policy_gather_ids_key: str = None,
        ref_gather_ids_key: str = None,
        skip_ref: bool = False,
    ):
        """Compute log-probs for a flat sample list."""
        assert not self.is_critic_model
        assert samples

        log_prob_top_k = getattr(self.ppo_config, "log_prob_top_k", 0)
        return_per_token_entropy = getattr(self.ppo_config, "loss_func", None) == "steer"

        if self.policy_config.smart_pad_infer:
            assert log_prob_top_k == 0, (
                "policy.smart_pad_infer + ppo.log_prob_top_k > 0 is not supported yet; "
                "disable smart_pad_infer for the OPD top-K path."
            )
            call_logps_func = self.smart_pad_compute_logprobs
            batch_log_str = "[smart_pad] get_{name} microbatch "
        else:
            call_logps_func = self.compute_logprobs
            batch_log_str = "get_{name} microbatch "

        has_ref = not self.policy_config.without_ref and not skip_ref
        policy_first = (policy_compute_topk and ref_gather_ids_key is not None and has_ref)
        assert len(samples) % self.forward_only_mbs == 0, (
            f"len(samples)={len(samples)} is not divisible by "
            f"forward_only_mbs={self.forward_only_mbs}; enable dynamic CP "
            f"or adjust policy.forward_only_mbs"
        )

        ref_logps = None
        prev_logps = None

        if policy_first:
            self.onload_model()
            for model_module in self.model:
                model_module.eval()
            policy_kw: Dict[str, Any] = {"compute_topk": True}
            if policy_gather_ids_key is not None:
                policy_kw["gather_target_ids_key"] = policy_gather_ids_key
            if return_per_token_entropy:
                policy_kw["return_per_token_entropy"] = True
            with self.get_router_replay_ctx():
                prev_logps = call_logps_func(
                    self.model,
                    samples,
                    batch_log_str=batch_log_str.format(name="policy_logprobs"),
                    **policy_kw,
                )
            for i, d in enumerate(prev_logps):
                samples[i][ref_gather_ids_key] = d["topk_ids"]

            self.offload_model()
            self.onload_ref_model()
            for model_module in self.ref_model:
                model_module.eval()
            ref_logps = call_logps_func(
                self.ref_model,
                samples,
                batch_log_str=batch_log_str.format(name="ref_policy_logprobs"),
                gather_target_ids_key=ref_gather_ids_key,
            )
            self.offload_ref_model()
            self.onload_model()
        else:
            if has_ref:
                self.offload_model()
                self.onload_ref_model()
                for model_module in self.ref_model:
                    model_module.eval()
                ref_logps = call_logps_func(
                    self.ref_model,
                    samples,
                    batch_log_str=batch_log_str.format(name="ref_policy_logprobs"),
                    **({
                        "gather_target_ids_key": ref_gather_ids_key
                    } if ref_gather_ids_key else {}),
                )
                self.offload_ref_model()

            if compute_pre_logps:
                self.onload_model()
                for model_module in self.model:
                    model_module.eval()
                policy_kw: Dict[str, Any] = {}
                if policy_compute_topk:
                    policy_kw["compute_topk"] = True
                if policy_gather_ids_key is not None:
                    policy_kw["gather_target_ids_key"] = policy_gather_ids_key
                if return_per_token_entropy:
                    policy_kw["return_per_token_entropy"] = True
                with self.get_router_replay_ctx():
                    prev_logps = call_logps_func(
                        self.model,
                        samples,
                        batch_log_str=batch_log_str.format(name="policy_logprobs"),
                        **policy_kw,
                    )
            else:
                self.onload_model()

        if self.policy_config.without_ref:
            assert ref_logps is None

        prev_per_token_entropies = None
        if return_per_token_entropy:
            assert prev_logps is not None
            prev_per_token_entropies = [result["prev_per_token_entropy"] for result in prev_logps]
            prev_logps = [result["logprobs"] for result in prev_logps]

        if compute_pre_logps:
            assert prev_logps is not None
            if return_per_token_entropy:
                assert prev_per_token_entropies is not None
        output = (ref_logps, prev_logps)
        if return_per_token_entropy:
            output += (prev_per_token_entropies, )
        return output

    def compute_log_probs_dynamic_cp(
        self,
        samples: List[Dict[str, Any]],
        compute_pre_logps: bool = True,
    ):
        """Compute log-probs with dynamic CP for a flat sample list."""
        assert not self.is_critic_model
        assert samples
        num_local_samples = len(samples)

        # Copy the list: rl_reroute replaces elements in-place with shifted
        # packing dicts. Callers still hold the original sample objects (e.g.
        # train_steps) and write logprobs back onto that same list.
        packed_batches, num_micro_batches, _, _, routing_info = (
            self.prepare_data.rl_reroute_data_for_dynamic_cp(
                list(samples),
                self.tokenizer.pad_token_id,
                vocab_size=self.vocab_size,
                pad_with_random_token=self.config.training.moe_pad_with_random_tokens,
                max_seqlen_per_dp_cp_rank=self.dist_config.max_seqlen_per_dp_cp_rank_fwd_only,
            )
        )
        self._assert_equal_num_microbatches(num_micro_batches)

        dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)
        global_ids_this_rank = routing_info["global_ids_this_rank"]
        global_id_logprob_lens = routing_info["global_id_logprob_lens"]
        gid_to_compute_rank = routing_info["gid_to_compute_rank"]
        gid_to_orig_dcp_rank = routing_info["gid_to_orig_dcp_rank"]

        seqlen = self.dist_config.max_seqlen_per_dp_cp_rank_fwd_only

        ref_logps = None
        prev_logps = None

        if not self.policy_config.without_ref:
            self.offload_model()
            self.onload_ref_model()
            for model_module in self.ref_model:
                model_module.eval()
            per_sample_ref = self._forward_packed_batches_unified(
                self.ref_model,
                packed_batches,
                num_micro_batches,
                seqlen,
                batch_log_str="get_ref_policy_logprobs (dyn_cp) microbatch ",
            )
            ref_logps = self._reverse_and_collect(
                per_sample_ref,
                global_ids_this_rank,
                global_id_logprob_lens,
                gid_to_compute_rank,
                gid_to_orig_dcp_rank,
                dp_cp_group,
                num_local_samples,
            )
            self.offload_ref_model()

        if compute_pre_logps:
            self.onload_model()
            for model_module in self.model:
                model_module.eval()
            with self.get_router_replay_ctx():
                per_sample_prev = self._forward_packed_batches_unified(
                    self.model,
                    packed_batches,
                    num_micro_batches,
                    seqlen,
                    batch_log_str="get_policy_logprobs (dyn_cp) microbatch ",
                )
            prev_logps = self._reverse_and_collect(
                per_sample_prev,
                global_ids_this_rank,
                global_id_logprob_lens,
                gid_to_compute_rank,
                gid_to_orig_dcp_rank,
                dp_cp_group,
                num_local_samples,
            )
        else:
            self.onload_model()

        if self.policy_config.without_ref:
            assert ref_logps is None
        if compute_pre_logps:
            assert prev_logps is not None
        return ref_logps, prev_logps

    def _align_values_to_samples(
        self,
        samples: List[Dict[str, Any]],
        values: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Project critic values onto each sample's policy-logprob coordinates."""
        for i, (sample, value) in enumerate(zip(samples, values, strict=True)):
            logprob = sample.get("logprobs")
            if logprob is None:
                continue
            target_len = logprob.size(-1)
            value_len = value.size(-1)
            if value_len == target_len:
                continue
            if value_len > target_len:
                values[i] = value[:target_len].contiguous()
                continue

            def to_int(x):
                return int(x.item()) if hasattr(x, "item") else int(x)

            prompt_length = sample.get("prompt_lengths")
            sequence_length = sample.get("sequence_lengths")
            prefix_len = None
            response_len = None
            if prompt_length is not None:
                prefix_len = max(to_int(prompt_length) - 1, 0)
            if prompt_length is not None and sequence_length is not None:
                response_len = max(to_int(sequence_length) - to_int(prompt_length), 0)

            padded_prefix_len = target_len - value_len
            if (
                prefix_len is not None and padded_prefix_len > 0 and
                padded_prefix_len <= prefix_len and
                (response_len is None or value_len >= response_len)
            ):
                values[i] = torch.nn.functional.pad(
                    value,
                    (padded_prefix_len, 0),
                    value=0.0,
                ).contiguous()
                continue

            if (
                prefix_len is not None and (
                    (
                        value_len == target_len - prefix_len and
                        (response_len is None or value_len >= response_len)
                    ) or (
                        response_len is not None and value_len == response_len and
                        prefix_len + value_len <= target_len
                    )
                )
            ):
                right_pad = target_len - prefix_len - value_len
                values[i] = torch.nn.functional.pad(
                    value,
                    (prefix_len, right_pad),
                    value=0.0,
                ).contiguous()
                continue

            raise RuntimeError(
                "critic values shorter than policy logprobs after CP gather: "
                f"sample={i}, values_shape={value.shape}, logprobs_shape={logprob.shape}, "
                f"prompt_length={prompt_length}, sequence_length={sequence_length}, "
                f"padded_prefix_len={padded_prefix_len}"
            )
        return values

    def compute_values(self, samples: List[Dict[str, Any]]):
        """Returns critic-model values for a flat sample list."""
        assert self.is_critic_model
        assert samples
        assert len(samples) % self.forward_only_mbs == 0, (
            f"len(samples)={len(samples)} is not divisible by "
            f"forward_only_mbs={self.forward_only_mbs}; enable dynamic CP "
            f"or adjust policy.forward_only_mbs"
        )

        if self.policy_config.smart_pad_infer:
            self.onload_model()
            for model_module in self.model:
                model_module.eval()

            self.batch_iters = 0
            total_samples = len(samples)
            self.total_iters = total_samples // self.forward_only_mbs
            self.batch_log_str = "[smart_pad] get_values microbatch "
            self._smart_pad_current_model = self.model

            dynamic_mbs_target_seqlen = getattr(
                self.policy_config, 'dynamic_mbs_target_seqlen_fwd_only', None
            )
            dynamic_mbs_limit = getattr(self.policy_config, 'dynamic_mbs_limit_fwd_only', None)

            smart_pad_helper = CatedSmartPadInferHelper(samples, self.forward_only_mbs)
            get_seqlen_func = lambda sample: sample["tokens"].shape[-1]
            try:
                smart_pad_helper.gen_row_based_batches()
                smart_pad_helper.gen_extend_batches(get_seqlen_func)
                smart_pad_helper.gen_sorted_batches()
                smart_pad_helper.gen_smart_pad_batches(self.training_config.pad_to_mulitiple_of)
                smart_pad_helper.forward_per_seqlen_batches(
                    forward_step_wrapped_func=self._smart_pad_value_forward_step,
                    dynamic_mbs_target_seqlen=dynamic_mbs_target_seqlen,
                    dynamic_mbs_limit=dynamic_mbs_limit,
                    update_total_iters_callback=lambda total_steps:
                    setattr(self, 'total_iters', total_steps),
                )

                values = []
                values_list = smart_pad_helper.get_rowed_based_forward_results(
                    is_row_based_rets=True
                )
                if mpu.is_pipeline_last_stage():
                    for per_forward_step_results in values_list:
                        for value in per_forward_step_results:
                            values.append(value.cpu())
            finally:
                self._smart_pad_current_model = None

            values = BroadcastUtils.broadcast_object_within_pp(values)
            assert len(values) == total_samples, (
                f"len(values) expect {total_samples}, but get {len(values)}"
            )
            clear_memory()
        else:
            self.onload_model()
            for model_module in self.model:
                model_module.eval()
            values = self._compute_values(
                self.model,
                samples,
                batch_log_str="get_values microbatch ",
            )
            assert values is not None
        return self._align_values_to_samples(samples, values)

    def rl_train_actor(self, dataloader_iter):
        assert not self.is_critic_model
        self.set_model_train()
        dumped_metrics_per_ppo_step = [] if self.should_dump_metrics else None
        metrics = {}
        for batch in dataloader_iter:
            assert batch
            for model_chunk in self.model:
                model_chunk.zero_grad_buffer()
            self.optimizer.zero_grad()

            (
                self._step_global_batch_size,
                self._step_effective_global_batch_size,
                self._step_global_token_cnt,
            ) = self._compute_step_gbs_and_token_cnt(batch)
            dyn_cp_stats = None
            routing_info = None
            if self.config.policy.dist_config.dynamic_context_parallel:
                batch, num_microbatches, seqlen_sum, seqlen_sq_sum, routing_info = (
                    self.prepare_data.rl_reroute_data_for_dynamic_cp(
                        list(batch),
                        self.tokenizer.pad_token_id,
                        vocab_size=self.vocab_size,
                        pad_with_random_token=self.config.training.moe_pad_with_random_tokens,
                        need_routing_info=self.should_dump_metrics,
                    )
                )
                self._assert_equal_num_microbatches(num_microbatches)
                dyn_cp_stats = {
                    "policy/dyn_cp_seqlen_sum": seqlen_sum,
                    "policy/dyn_cp_seqlen_sq_sum": seqlen_sq_sum,
                    "policy/dyn_cp_num_micro_batches": num_microbatches,
                }
            else:
                num_microbatches = self._static_num_microbatches(batch)

            with self.get_router_replay_ctx():
                metric = self._update_policy(
                    batch,
                    num_microbatches=num_microbatches,
                    routing_info=routing_info,
                )
            if dyn_cp_stats is not None:
                metric.update(dyn_cp_stats)
            self._rl_collect_mb_dumped_metrics(metric, dumped_metrics_per_ppo_step)
            if clear_gathered_routing_info is not None:
                clear_gathered_routing_info()
            extend_value_to_dict(metrics, metric)

            update_successful, grad_norm, _ = self.optimizer.step()
            extend_value_to_dict(metrics, {"policy/grad_norm": grad_norm})
            post_clip_grad_norm = self.maybe_post_clip_grad_norm(update_successful)
            if post_clip_grad_norm is not None:
                extend_value_to_dict(
                    metrics,
                    {"policy/post_clip_grad_norm": post_clip_grad_norm},
                )
            if not update_successful:
                raise RuntimeError("Optimizer step failed")
            if self.config.optimizer.update_lr_by_train_step:
                self.optimizer_scheduler.step(1)
            if self.ppo_config.use_adaptive_entropy:
                assert "policy/scaled_entropy" in metric, (
                    "use_adaptive_entropy requires policy/scaled_entropy in train metrics"
                )
                extend_value_to_dict(
                    metrics,
                    update_adaptive_entropy_after_train_step(
                        self.ppo_config,
                        float(metric["policy/scaled_entropy"]),
                    ),
                )

        if dumped_metrics_per_ppo_step:
            metrics["dumped_metrics_per_ppo_step"] = dumped_metrics_per_ppo_step
        clear_memory()
        return metrics

    def rl_train_value(self, dataloader_iter):
        assert self.is_critic_model
        assert not self.policy_config.dist_config.dynamic_context_parallel, (
            "dynamic-batch critic training does not support dynamic CP"
        )
        self.set_model_train()
        metrics = {}
        for batch in dataloader_iter:
            assert batch
            for model_chunk in self.model:
                model_chunk.zero_grad_buffer()
            self.optimizer.zero_grad()
            num_microbatches = self._static_num_microbatches(batch)
            metric = self._update_value_model(batch, num_microbatches=num_microbatches)
            extend_value_to_dict(metrics, metric)

            update_successful, grad_norm, _ = self.optimizer.step()
            extend_value_to_dict(metrics, {"value/grad_norm": grad_norm})
            post_clip_grad_norm = self.maybe_post_clip_grad_norm(update_successful)
            if post_clip_grad_norm is not None:
                extend_value_to_dict(
                    metrics,
                    {"value/post_clip_grad_norm": post_clip_grad_norm},
                )
            if not update_successful:
                raise RuntimeError("Optimizer step failed")
            if self.config.optimizer.update_lr_by_train_step:
                self.optimizer_scheduler.step(1)
        clear_memory()
        return metrics
