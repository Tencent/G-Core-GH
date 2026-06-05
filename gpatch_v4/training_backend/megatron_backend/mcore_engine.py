import os
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed
from transformers import AutoTokenizer
from typing_extensions import override

from megatron.core import mpu
from megatron.core.optimizer.optimizer import (
    ChainedOptimizer,
    FP32Optimizer,
    MixedPrecisionOptimizer,
)

from gpatch_v4.core import parallel_state
from gpatch_v4.core.parallel_state import is_tp_and_cp_head
from gpatch_v4.extended_model import PrepareDataForwardFactory
from gpatch_v4.training_backend.base_engine import BaseEngine
from gpatch_v4.training_backend.common.swap_mixin import EngineSwapMixin
from gpatch_v4.training_backend.megatron_backend.checkpoint import (
    get_latest_checkpoint_folder,
    load_checkpoint,
)
from gpatch_v4.training_backend.megatron_backend.mcore_swap_impl import McoreSwapImpl
from gpatch_v4.training_backend.megatron_backend.megatron_utils import unwrap_model
from gpatch_v4.training_backend.megatron_backend.mixin import (
    BridgeUtilsMixin,
    CheckpointMixin,
    ForwardStepMixin,
)
from gpatch_v4.training_backend.megatron_backend.optimizer import (
    get_megatron_last_lr,
    get_optimizer_and_scheduler,
)
from gpatch_v4.utils import (
    cpu_dict,
    expand_rollout_batches,
    extend_value_to_dict,
    log,
    log_info,
    logging_memory_usage,
    logging_memory_usage_details,
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
    sync_cuda_and_get_time,
)
from gpatch_v4.utils.common_utils import (
    clear_memory,
    logging_rank0,
    profile_memory_and_time,
)
from gpatch_v4.utils.training_utils import get_dump_moe_metrics

try:
    from megatron.core.gcore_utils import (
        clear_gathered_routing_info,  # only wxdev support
    )
except ImportError:
    clear_gathered_routing_info = None

# Optimizer types whose `get_grad_norm()` is known to share the same
# reduction topology and main-grad buffer with the internal clip_grad_norm.
# Other types (LayerWiseOptimizer / muon / etc.) skip with warn-once.
POST_CLIP_OK_OPT_TYPES = (MixedPrecisionOptimizer, FP32Optimizer)


def is_post_clip_grad_norm_supported(optimizer) -> bool:
    if isinstance(optimizer, POST_CLIP_OK_OPT_TYPES):
        return True
    if isinstance(optimizer, ChainedOptimizer):
        return all(isinstance(o, POST_CLIP_OK_OPT_TYPES) for o in optimizer.chained_optimizers)
    return False


class McoreEngine(BaseEngine, BridgeUtilsMixin, EngineSwapMixin, ForwardStepMixin, CheckpointMixin):
    def __init__(self, config, policy_config, tokenizer: AutoTokenizer, is_critic_model=False):
        super().__init__(config, policy_config, tokenizer)

        self.swap_impl = McoreSwapImpl()
        self.forward_only_mbs = self.policy_config.forward_only_mbs
        self.prepare_data = PrepareDataForwardFactory.get_prepare_data_fwd(config)

        ctx = self.disable_moe_router_replay() if is_critic_model else nullcontext()
        with ctx:
            self.bridge, self.hf_config = self.build_bridge(
                self.policy_config.hf_model_path,
                override_transformer_config=self.policy_config.override_transformer_config
            )
        self.is_critic_model = is_critic_model
        if not self.policy_config.without_ref:
            # disable moe router replay for ref model
            with self.disable_moe_router_replay():
                self.ref_bridge, _ = self.build_bridge(
                    self.policy_config.ref_hf_model_path,
                    override_transformer_config=self.policy_config.override_transformer_config
                )
        self.logits_cpu_buffer = None

        #TODO: 如果是 early swap actor model，看看是否需要创建一个 cpu_model_dict

    def setup_ref_model(self):
        save_latest_step = get_latest_checkpoint_folder(self.checkpoint_config.save_ref_ckpt_path)
        load_latest_step = get_latest_checkpoint_folder(self.checkpoint_config.load_ref_ckpt_path)
        load_weights_from_mbridge = save_latest_step is None and load_latest_step is None

        self.ref_model, self.ref_mcore_config = self.get_model_from_bridge(
            self.ref_bridge,
            self.policy_config.ref_hf_model_path,
            load_weights_from_mbridge=load_weights_from_mbridge,
            model_type=f"ref model",
            wrap_with_ddp=False,
        )
        if not load_weights_from_mbridge:
            # 如果不用 mbrige 转出来权重
            #TODO: 从 load_ref_ckpt_path or load_ref_ckpt_path 加载
            raise NotImplementedError("Not implemented yet")
        with profile_memory_and_time(f"offload_ref_model", rank=0):
            self.offload_ref_model()

    @override
    def setup_model_and_get_optimizer(self):
        logging_memory_usage_details("memory tracking before setup_model_and_get_optimizer", rank=0)
        # build and offload reference model
        if self.policy_config.without_ref:
            self.ref_model = None
            self.get_swap_state().ref_model = False
        else:
            assert not self.is_critic_model
            # disable moe router replay for ref model
            with self.disable_moe_router_replay():
                self.setup_ref_model()

        load_latest_step = get_latest_checkpoint_folder(self.checkpoint_config.load_ckpt_path)
        load_weights_from_mbridge = load_latest_step is None
        ctx = self.disable_moe_router_replay() if self.is_critic_model else nullcontext()
        with ctx:
            self.model, self.mcore_config = self.get_model_from_bridge(
                self.bridge,
                self.policy_config.hf_model_path,
                load_weights_from_mbridge=load_weights_from_mbridge,
                model_type=f"policy model",
                wrap_with_ddp=self.policy_config.wrap_with_ddp,
                build_value_model=self.is_critic_model,
            )

        if self.config.training.moe_router_replay:
            # RouterReplay 实例是进程级全局状态，policy model 和 ref model
            # 的实例会混在一起。这里从 policy model 的模块树中显式收集
            # RouterReplay 实例，确保后续 replay 操作只作用于 policy model。
            # unwrap_model 可能返回 list（pipeline parallel）或单个 module。
            unwrapped = unwrap_model(self.model)
            target = unwrapped[0] if isinstance(unwrapped, list) else unwrapped
            self.get_router_replay_manager().set_model_instances(target)

        log(f"mcore_engine config {self.mcore_config=} {self.hf_config=}", rank=0)
        unwrapped_model = unwrap_model(self.model)

        if hasattr(unwrapped_model[0], "vocab_size"):
            self.vocab_size = unwrapped_model[0].vocab_size
        else:
            # multimodal
            self.vocab_size = unwrapped_model[0].language_model.vocab_size
        prev_ppo_step = 0
        if self.policy_config.without_optim:
            return prev_ppo_step

        optimizer, optimizer_scheduler = get_optimizer_and_scheduler(self.config, self.model)
        if load_weights_from_mbridge:
            self.optimizer, self.optimizer_scheduler = optimizer, optimizer_scheduler
        else:
            prev_ppo_step = load_checkpoint(
                self.config,
                self.model,
                optimizer,
                optimizer_scheduler,
                load_latest_step,
                bridge=self.bridge
            )
            self.optimizer, self.optimizer_scheduler = optimizer, optimizer_scheduler

        if self.optimizer is not None:
            self.mcore_config.grad_scale_func = self.optimizer.scale_loss
        return prev_ppo_step

    def step_and_get_lr(self):
        if not self.config.optimizer.update_lr_by_train_step:
            self.optimizer_scheduler.step(1)
        lr = get_megatron_last_lr(self.optimizer)
        return lr

    def maybe_post_clip_grad_norm(self, update_successful: bool) -> Optional[float]:
        """Return post-clip total L2 grad norm, or None if disabled/unsupported.

        Must be called after ``self.optimizer.step()`` and before the next
        batch's ``zero_grad_buffer`` / ``zero_grad`` (main_grads still hold
        the clipped gradients). Reuses ``optimizer.get_grad_norm()`` so the
        reduction topology matches mcore's internal pre-clip clip_grad_norm.
        """
        if not self.config.optimizer.report_post_clip_grad_norm:
            return None
        if not update_successful:
            return None
        if self.optimizer.is_stub_optimizer:
            # Skip: grad_stats_parallel_group may differ from non-stub ranks.
            return None
        # mxfp8 path may overwrite main_grads with param data inside step().
        assert not self.optimizer.config.reuse_grad_buf_for_mxfp8_param_ag, (
            "report_post_clip_grad_norm is incompatible with "
            "reuse_grad_buf_for_mxfp8_param_ag."
        )
        if not is_post_clip_grad_norm_supported(self.optimizer):
            if not getattr(self, "_warned_post_clip_unsupported_opt", False):
                log_info(
                    f"WARN: report_post_clip_grad_norm: optimizer type "
                    f"{type(self.optimizer).__name__} is outside the validated "
                    f"whitelist; post-clip grad_norm will not be reported.",
                    rank=0,
                )
                self._warned_post_clip_unsupported_opt = True
            return None
        return self.optimizer.get_grad_norm()

    def export_weights(self):
        if self.config.training.build_from_mbridge:
            yield from self.bridge.export_weights(self.model)
        else:
            yield from self.bridge.export_hf_weights(self.model, show_progress=False)

    @override
    def compute_log_probs(
        self,
        rollout_batches: List[Dict[str, List[Any]]],
        compute_pre_logps=True,
        topk_gather_ids_key: str = None,
        ref_topk_gather_ids_key: str = None,
    ):
        """计算 ref model 和 actor model 的 log-probs。

        Args:
            topk_gather_ids_key: 用于 teacher engine —— 从 batch dict 取该 key
                对应的 ``[S-1, K]`` token-id tensor，在这些 ID 上 gather log-probs。
            ref_topk_gather_ids_key: 用于 G-OPD student engine —— 交换 student/ref
                forward 顺序，先跑 student 产出 topk_ids，再让 ref 在这些 ID 上
                gather log-probs。
            两参数互斥。

        Returns:
            ref_logps: ref model 的 log-probs。设置 ``ref_topk_gather_ids_key``
                时返回 ``{"logprobs", "gather_logprobs"}`` dict list，否则为
                ref_logps 2D tensor list。
            prev_logps: actor model 的 prev-step log-probs。``log_prob_top_k > 0``
                时返回 dict list（含 3D topk 字段），否则为 prev_logps 2D tensor list。
        """
        assert not self.is_critic_model

        log_prob_top_k = getattr(self.ppo_config, "log_prob_top_k", 0)
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

        samples_per_batch = len(rollout_batches[0]['tokens'])
        batches_list = expand_rollout_batches(rollout_batches)
        assert len(batches_list) == samples_per_batch * len(rollout_batches)

        # G-OPD 在 log_prob_top_k>0 且 without_ref False时，ref_logps需要用
        # student forward 的 topk-ids 来拿到 base_on_stu_topk_logprobs.
        need_ref_gather_swap = (
            ref_topk_gather_ids_key is not None and log_prob_top_k > 0 and
            not self.policy_config.without_ref and compute_pre_logps
        )
        assert not (need_ref_gather_swap and topk_gather_ids_key is not None), (
            "topk_gather_ids_key (teacher path) cannot be combined with "
            "ref_topk_gather_ids_key (student-engine ref-gather path)."
        )

        ref_logps = None
        prev_logps = None
        if need_ref_gather_swap:
            # Student forward first so its top-K ids can drive the ref gather.
            self.onload_model()
            for model_module in self.model:
                model_module.eval()
            with self.get_router_replay_ctx():
                prev_logps = call_logps_func(
                    self.model,
                    batches_list,
                    batch_log_str=batch_log_str.format(name="policy_logprobs"),
                    compute_topk=True,
                )
            # prev_logps is a list of per-sample dicts containing "topk_ids".
            for i, d in enumerate(prev_logps):
                batches_list[i][ref_topk_gather_ids_key] = d["topk_ids"]

            self.offload_model()
            self.onload_ref_model()
            for model_module in self.ref_model:
                model_module.eval()
            ref_logps = call_logps_func(
                self.ref_model,
                batches_list,
                batch_log_str=batch_log_str.format(name="ref_policy_logprobs"),
                gather_target_ids_key=ref_topk_gather_ids_key,
            )
            self.offload_ref_model()
            self.onload_model()
        else:
            if not self.policy_config.without_ref:
                self.offload_model()
                self.onload_ref_model()
                for model_module in self.ref_model:
                    model_module.eval()
                ref_logps = call_logps_func(
                    self.ref_model,
                    batches_list,
                    batch_log_str=batch_log_str.format(name="ref_policy_logprobs"),
                )
                self.offload_ref_model()

            if compute_pre_logps:
                self.onload_model()
                for model_module in self.model:
                    model_module.eval()
                # maybe enable r3 replay if required
                prev_kw: Dict[str, Any] = {}
                if log_prob_top_k > 0:
                    if topk_gather_ids_key is not None:
                        prev_kw["gather_target_ids_key"] = topk_gather_ids_key
                    else:
                        prev_kw["compute_topk"] = True
                with self.get_router_replay_ctx():
                    prev_logps = call_logps_func(
                        self.model,
                        batches_list,
                        batch_log_str=batch_log_str.format(name="policy_logprobs"),
                        **prev_kw,
                    )
            elif not self.policy_config.without_ref:
                # Policy was offloaded for ref forward; re-onload for training.
                self.onload_model()

        # 这里后面就要训练了，应该是不需要多 offload 一次

        def restor_shape(logps):
            if logps is None:
                return None
            res = []
            bs = len(logps) // samples_per_batch
            assert bs * samples_per_batch == len(logps)
            for i in range(bs):
                res.append(logps[i * samples_per_batch:(i + 1) * samples_per_batch])
            return res

        ref_logps_list = restor_shape(ref_logps)
        if self.policy_config.without_ref:
            assert ref_logps_list is None

        prev_logps_list = None
        if compute_pre_logps:
            prev_logps_list = restor_shape(prev_logps)
            assert prev_logps_list is not None
        return ref_logps_list, prev_logps_list

    def _rl_collect_mb_dumped_metrics(self, _metric, dumped_metrics_per_ppo_step):
        """each microbatch collect dumped loss_fn and moe metrics"""
        dumped_loss_fn_metrics = _metric.pop("dumped_loss_fn_metrics", None)
        if dumped_metrics_per_ppo_step is None or dumped_loss_fn_metrics is None:
            return
        dump_moe_topk = getattr(self.training_config, "ppo_dump_moe_topk", 0) or 0
        if dump_moe_topk > 0:
            dumped_moe_topk_metrics = None
            if is_tp_and_cp_head():
                dumped_moe_topk_metrics = get_dump_moe_metrics(
                    is_full_recompute=getattr(self.mcore_config, "recompute_granularity",
                                              None) == "full",
                    num_samples=len(dumped_loss_fn_metrics),
                )
            if dumped_moe_topk_metrics is not None and mpu.is_pipeline_first_stage():
                assert len(dumped_loss_fn_metrics) <= len(dumped_moe_topk_metrics), \
                    f"dumped_loss_fn_metrics length {len(dumped_loss_fn_metrics)} > dumped_moe_topk_metrics length {len(dumped_moe_topk_metrics)}"
                # 由于dumped_moe_topk_metrics是training阶段采集的数据，当开启moe_layer_recompute或者recompute_granularity时，
                # dumped_moe_topk_metrics的长度可能会大于all_dumped_metrics的长度，取前len(all_dumped_metrics)个就好了。
                for s, topk in zip(
                    dumped_loss_fn_metrics, dumped_moe_topk_metrics[:len(dumped_loss_fn_metrics)]
                ):
                    s["moe_topk_info"] = topk
        dumped_metrics_per_ppo_step.extend(dumped_loss_fn_metrics)

    @override
    def rl_train_actor(
        self,
        dataloader_iter,
    ):
        assert not self.is_critic_model
        num_microbatches = self.training_config.train_gbs // (
            self.training_config.train_mbs * mpu.get_data_parallel_world_size()
        )
        self.set_model_train()

        dumped_metrics_per_ppo_step = [] if self.should_dump_metrics else None
        metrics = {}
        for batch in dataloader_iter:
            for model_chunk in self.model:
                model_chunk.zero_grad_buffer()
            self.optimizer.zero_grad()
            # maybe enable r3 replay if required
            with self.get_router_replay_ctx():
                _metric = self._update_policy(batch, num_microbatches=num_microbatches)

            self._rl_collect_mb_dumped_metrics(_metric, dumped_metrics_per_ppo_step)

            if clear_gathered_routing_info is not None:
                clear_gathered_routing_info()

            extend_value_to_dict(
                metrics, _metric
            )  # append the metric from this micro-batch to global metrics.

            update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()
            extend_value_to_dict(metrics, {"policy/grad_norm": grad_norm})
            post_clip_grad_norm = self.maybe_post_clip_grad_norm(update_successful)
            if post_clip_grad_norm is not None:
                extend_value_to_dict(metrics, {"policy/post_clip_grad_norm": post_clip_grad_norm})

            if update_successful:
                if self.config.optimizer.update_lr_by_train_step:
                    self.optimizer_scheduler.step(1)
            else:
                raise RuntimeError("Optimizer step failed")

        if dumped_metrics_per_ppo_step:
            metrics["dumped_metrics_per_ppo_step"] = dumped_metrics_per_ppo_step
        clear_memory()
        return metrics

    @override
    def finetune_step(self, batch: List[Dict[str, Any]], num_microbatches: int, step: int):
        for model_chunk in self.model:
            model_chunk.zero_grad_buffer()
        self.optimizer.zero_grad()

        if self.config.policy.dist_config.dynamic_context_parallel:
            assert not self.config.training.use_dynamic_mbs
            batch, num_microbatches, seqlen_sum, seqlen_sq_sum = (
                self.prepare_data.sft_reroute_data_for_dynamic_cp(
                    batch,
                    self.tokenizer.pad_token_id,
                    vocab_size=self.vocab_size,
                    pad_with_random_token=self.config.training.moe_pad_with_random_tokens,
                )
            )

        metric = self._finetune_step(batch, num_microbatches=num_microbatches)
        update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()
        post_clip_grad_norm = self.maybe_post_clip_grad_norm(update_successful)
        lr = self.step_and_get_lr()

        update_successful = logical_and_across_model_parallel_group(update_successful)
        # grad_norm and num_zeros_in_grad will be None on ranks without trainable params,
        # so we must gather across mp ranks
        grad_norm = reduce_max_stat_across_model_parallel_group(grad_norm)
        num_zeros_in_grad = reduce_max_stat_across_model_parallel_group(num_zeros_in_grad)

        metric["finetune/grad_norm"] = grad_norm
        metric["finetune/lr"] = lr
        metric["finetune/num_zeros_in_grad"] = num_zeros_in_grad
        if post_clip_grad_norm is not None:
            # Mirror grad_norm's reduce_max for ranks-without-trainable-params consistency.
            metric["finetune/post_clip_grad_norm"] = (
                reduce_max_stat_across_model_parallel_group(post_clip_grad_norm)
            )
        if self.config.policy.dist_config.dynamic_context_parallel:
            metric["finetune/dyn_cp_seqlen_sum"] = seqlen_sum
            metric["finetune/dyn_cp_seqlen_sq_sum"] = seqlen_sq_sum
            metric["finetune/dyn_cp_num_micro_batches"] = num_microbatches

        if self.policy_config.manual_clear_memory and (
            step + 1
        ) % self.policy_config.manual_clear_memory_interval == 0:
            clear_memory()
        return metric

    @torch.no_grad()
    def eval_step(self, batch: List[Dict[str, Any]], num_microbatches: int):
        metric = self._finetune_step(batch, num_microbatches=num_microbatches, forward_only=True)
        clear_memory()
        return metric

    def compute_logits(self, rollout_batches: List[Dict[str, torch.Tensor]]):
        if self.policy_config.smart_pad_infer:
            return self.smart_pad_compute_logits(
                self.model,
                rollout_batches,
                batch_log_str="[smart_pad] get_policy_logits microbatch ",
            )
        else:
            return self.default_compute_logits(rollout_batches)

    def default_compute_logits(self, rollout_batches: List[Dict[str, torch.Tensor]]):
        begine_t = sync_cuda_and_get_time()
        assert rollout_batches[0]['tokens'].ndim == 1, f"{rollout_batches[0]['tokens'].ndim}"

        assert self.policy_config.without_ref

        self.set_model_eval()
        logits_output = self._compute_logits(
            self.model,
            rollout_batches,
            batch_log_str="get_policy_logits microbatch ",
        )
        end_t = sync_cuda_and_get_time()

        if mpu.is_pipeline_last_stage():
            assert logits_output is not None
            log(f"default_compute_logits using time {end_t - begine_t}", rank=0)
        else:
            logits_output = [None for _ in range(len(rollout_batches))]
        return None, logits_output

    @override
    def set_model_eval(self):
        for model_module in self.model:
            model_module.eval()

    @override
    def set_model_train(self):
        for model_module in self.model:
            model_module.train()

    def normal_compute_values(self, rollout_batches: List[Dict[str, List[Any]]]):
        samples_per_batch = len(rollout_batches[0]['tokens'])
        batches_list = expand_rollout_batches(rollout_batches)
        assert len(batches_list) == samples_per_batch * len(rollout_batches)
        self.onload_model()
        for model_module in self.model:
            model_module.eval()

        values = self._compute_values(
            self.model,
            batches_list,
            batch_log_str="get_values microbatch ",
        )

        # 这里后面就要训练了，应该是不需要多 offload 一次
        def restor_shape(values):
            if values is None:
                return None
            res = []
            bs = len(values) // samples_per_batch
            assert bs * samples_per_batch == len(values)
            for i in range(bs):
                res.append(values[i * samples_per_batch:(i + 1) * samples_per_batch])
            return res

        values_list = restor_shape(values)
        assert values_list is not None
        return values_list

    def smart_pad_compute_values(self, rollout_batches: List[Dict[str, List[Any]]]):
        raise NotImplementedError("Not implemented yet")

    def compute_values(self, rollout_batches: List[Dict[str, List[Any]]]):
        """Returns critic-model values."""
        assert self.is_critic_model
        if self.policy_config.smart_pad_infer:
            return self.smart_pad_compute_values(rollout_batches)
        else:
            return self.normal_compute_values(rollout_batches)

    def rl_train_value(
        self,
        dataloader_iter,
    ):
        assert self.is_critic_model
        num_microbatches = self.training_config.train_gbs // (
            self.training_config.train_mbs * mpu.get_data_parallel_world_size()
        )
        self.set_model_train()

        metrics = {}
        for batch in dataloader_iter:
            for model_chunk in self.model:
                model_chunk.zero_grad_buffer()
            self.optimizer.zero_grad()
            # maybe enable r3 replay if required
            _metric = self._update_value_model(batch, num_microbatches=num_microbatches)

            extend_value_to_dict(
                metrics, _metric
            )  # append the metric from this micro-batch to global metrics.

            update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()
            extend_value_to_dict(metrics, {"value/grad_norm": grad_norm})
            post_clip_grad_norm = self.maybe_post_clip_grad_norm(update_successful)
            if post_clip_grad_norm is not None:
                extend_value_to_dict(metrics, {"value/post_clip_grad_norm": post_clip_grad_norm})

            if update_successful:
                if self.config.optimizer.update_lr_by_train_step:
                    self.optimizer_scheduler.step(1)
            else:
                raise RuntimeError("Optimizer step failed")
        clear_memory()
        return metrics
