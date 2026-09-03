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
from megatron.core.pipeline_parallel import get_forward_backward_func

from gpatch_v4.configs.config import RewardConfig
from gpatch_v4.core import parallel_state
from gpatch_v4.core.adaptive_entropy import update_adaptive_entropy_after_train_step
from gpatch_v4.core.mappings import all_gather_from_context_parallel_region
from gpatch_v4.core.parallel_state import is_tp_and_cp_head
from gpatch_v4.core.smart_pad_helper import CatedSmartPadInferHelper
from gpatch_v4.extended_model import PrepareDataForwardFactory
from gpatch_v4.training_backend.base_engine import BaseEngine
from gpatch_v4.training_backend.common.swap_mixin import EngineSwapMixin
from gpatch_v4.training_backend.megatron_backend.checkpoint import (
    get_latest_checkpoint_folder,
    load_checkpoint,
)
from gpatch_v4.training_backend.megatron_backend.mcore_peft import is_peft_enabled
from gpatch_v4.training_backend.megatron_backend.mcore_swap_impl import McoreSwapImpl
from gpatch_v4.training_backend.megatron_backend.megatron_utils import unwrap_model
from gpatch_v4.training_backend.megatron_backend.mixin import (
    BridgeUtilsMixin,
    CheckpointMixin,
    ForwardStepMixin,
    MetricMixin,
)
from gpatch_v4.training_backend.megatron_backend.optimizer import (
    get_megatron_last_lr,
    get_optimizer_and_scheduler,
)

try:
    from gpatch_v4.training_backend.megatron_backend.welm_v45_myfa import (
        install_welm_v45_myfa_hooks,
    )
except:
    install_welm_v45_myfa_hooks = None

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
from gpatch_v4.utils.communication_utils import BroadcastUtils
from gpatch_v4.utils.dynamic_cp_utils import reverse_reroute_logprobs
from gpatch_v4.utils.training_utils import (
    from_parallel_logits_to_logprobs,
    get_dump_moe_metrics,
    get_scale_as_float,
    logprobs_from_linear_ce,
)

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


def _propagate_fp8_overrides(policy_config, optimizer_config=None) -> None:
    """``fp8_param_gather`` -> ``fp8_param`` + ddp/optimizer (Megatron-aligned)."""
    tc = policy_config.override_transformer_config
    if not tc.get("fp8"):
        return

    gather = bool(tc.pop("fp8_param_gather", False))
    tc["fp8_param"] = gather
    policy_config.override_ddp_config["fp8_param_gather"] = gather

    if optimizer_config is None:
        return
    if optimizer_config.override_optimizer_config is None:
        optimizer_config.override_optimizer_config = {}
    optim = optimizer_config.override_optimizer_config
    if "fp8_recipe" in tc:
        optim["fp8_recipe"] = tc["fp8_recipe"]
    if gather:
        optim["use_precision_aware_optimizer"] = True


class McoreEngine(
    BaseEngine, BridgeUtilsMixin, EngineSwapMixin, ForwardStepMixin, CheckpointMixin, MetricMixin
):
    def __init__(self, config, policy_config, tokenizer: AutoTokenizer, is_critic_model=False):
        super().__init__(config, policy_config, tokenizer)
        _propagate_fp8_overrides(
            self.policy_config,
            optimizer_config=getattr(self.config, "optimizer", None),
        )
        if self.policy_config.use_megatron_fsdp:
            assert not self.config.training.build_from_mbridge, (
                "policy.use_megatron_fsdp=True requires training.build_from_mbridge=False "
                "so gpatch_v4 uses the Megatron-Bridge path."
            )
            assert self.policy_config.wrap_with_ddp, (
                "policy.use_megatron_fsdp=True requires policy.wrap_with_ddp=True."
            )

        early_swap_model = getattr(self.config.training, "early_swap_model", False)
        if early_swap_model:
            assert mpu.get_pipeline_model_parallel_world_size(
            ) == 1, ("McoreEngine early_swap_model does not support pipeline parallelism")

        self.swap_impl = McoreSwapImpl(early_swap_model=early_swap_model)
        self.forward_only_mbs = self.policy_config.forward_only_mbs
        self.prepare_data = PrepareDataForwardFactory.get_prepare_data_fwd(config)

        ctx = self.disable_moe_router_replay() if is_critic_model else nullcontext()
        with ctx:
            self.bridge, self.hf_config = self.build_bridge(
                self.policy_config.hf_model_path,
                override_transformer_config=self.policy_config.override_transformer_config
            )
        self.is_critic_model = is_critic_model
        # Reward-model training reuses the scalar value head (LinearForLastLayer).
        self.build_reward_head = isinstance(config, RewardConfig)
        self.peft = None
        if not self.policy_config.without_ref:
            # disable moe router replay for ref model
            with self.disable_moe_router_replay():
                self.ref_bridge, self.ref_hf_config = self.build_bridge(
                    self.policy_config.ref_hf_model_path,
                    override_transformer_config=self.policy_config.override_transformer_config
                )
        self.logits_cpu_buffer = None
        self.teacher_output_weight = None

    @torch.no_grad()
    def setup_teacher_output_weight(self) -> None:
        """Load a frozen Teacher lm-head shard on Student PP-last ranks.

        The Teacher bridge mapping is used to resolve the architecture-specific
        MCore output-layer name to its Hugging Face checkpoint key. Each TP
        rank loads the full HF tensor and selects its vocabulary shard.
        """
        if not mpu.is_pipeline_last_stage():
            self.teacher_output_weight = None
            return

        teacher_bridge, _ = self.build_bridge(
            self.config.teacher.hf_model_path,
            override_transformer_config=self.config.teacher.override_transformer_config,
        )
        assert not getattr(teacher_bridge.config, "use_mup", False), (
            "Teacher hidden-state transfer does not yet support MuP logit scaling"
        )
        output_mappings = [
            (mcore_name, hf_name) for mcore_name, hf_name in teacher_bridge._DIRECT_MAPPING.items()
            if mcore_name.endswith("output_layer.weight")
        ]
        assert len(output_mappings) == 1, (
            "Expected exactly one Teacher output-layer entry in mbridge _DIRECT_MAPPING, "
            f"got {output_mappings}"
        )
        mcore_name, hf_name = output_mappings[0]
        io = teacher_bridge._get_safetensor_io(self.config.teacher.hf_model_path)

        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        assert self.vocab_size % tp_size == 0
        hf_weight = io.load_one_hf_weight(hf_name)
        full_weight = teacher_bridge._weight_to_mcore_format(mcore_name, [hf_weight])
        assert full_weight.ndim == 2
        assert full_weight.shape[0] == self.vocab_size, (
            "Teacher vocab is larger than the Student padded vocab: "
            f"{full_weight.shape[0]} vs {self.vocab_size}"
        )
        self.teacher_output_weight = (
            full_weight.chunk(tp_size, dim=0)[tp_rank].to(
                device=torch.cuda.current_device(), dtype=torch.bfloat16
            ).contiguous().detach()
        )
        logging_rank0(
            "Loaded frozen Teacher output-layer shard "
            f"{tuple(self.teacher_output_weight.shape)} from {hf_name}"
        )

    def setup_ref_model(self):
        save_latest_step = get_latest_checkpoint_folder(self.checkpoint_config.save_ref_ckpt_path)
        load_latest_step = get_latest_checkpoint_folder(self.checkpoint_config.load_ref_ckpt_path)
        load_weights_from_bridge = save_latest_step is None and load_latest_step is None

        self.ref_model, self.ref_mcore_config = self.get_model_from_bridge(
            self.ref_bridge,
            self.policy_config.ref_hf_model_path,
            load_weights_from_bridge=load_weights_from_bridge,
            model_type=f"ref model",
            wrap_with_ddp=False,
        )

        installed = (install_welm_v45_myfa_hooks is not None
                    ) and install_welm_v45_myfa_hooks(self.ref_model, self.ref_hf_config)
        if installed:
            log(f"installed WeLM v4.5 MyFA hooks on ref model: {installed}", rank=0)
        if not load_weights_from_bridge:
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
        has_peft = is_peft_enabled(self.policy_config)
        # Only skip HF load when this engine will later restore weights from dist
        # checkpoint. Teacher / other without_optim roles never call
        # load_checkpoint; if we also skip HF load whenever a student ckpt
        # exists, they keep random-init weights (OPD resume: teacher_kl ~ log V).
        will_load_dist_ckpt = (
            load_latest_step is not None and not self.policy_config.without_optim
        )
        # PEFT checkpoints store adapters only; base weights always come from HF.
        load_hf_base_weights = (not will_load_dist_ckpt) or has_peft
        ctx = self.disable_moe_router_replay() if self.is_critic_model else nullcontext()
        with ctx:
            self.model, self.mcore_config = self.get_model_from_bridge(
                self.bridge,
                self.policy_config.hf_model_path,
                load_weights_from_bridge=load_hf_base_weights,
                model_type=f"policy model",
                wrap_with_ddp=self.policy_config.wrap_with_ddp,
                build_value_model=self.is_critic_model or self.build_reward_head,
            )
        installed = (install_welm_v45_myfa_hooks
                     is not None) and install_welm_v45_myfa_hooks(self.model, self.hf_config)
        if installed:
            log(f"installed WeLM v4.5 MyFA hooks on policy model: {installed}", rank=0)
        logging_memory_usage_details("memory tracking after policy model load", rank=0)

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
            logging_memory_usage_details(
                "memory tracking after setup_model_and_get_optimizer", rank=0
            )
            return prev_ppo_step

        optimizer, optimizer_scheduler = get_optimizer_and_scheduler(self.config, self.model)
        if load_latest_step is None:
            prev_ppo_step = 0
            self.optimizer, self.optimizer_scheduler = optimizer, optimizer_scheduler
        else:
            prev_ppo_step = load_checkpoint(
                self.config,
                self.model,
                optimizer,
                optimizer_scheduler,
                load_latest_step,
                bridge=self.bridge,
                peft=self.peft,
                use_megatron_fsdp=self.policy_config.use_megatron_fsdp,
            )
            self.optimizer, self.optimizer_scheduler = optimizer, optimizer_scheduler

        if self.optimizer is not None:
            self.mcore_config.grad_scale_func = self.optimizer.scale_loss
        clear_memory()
        logging_memory_usage_details("memory tracking after setup_model_and_get_optimizer", rank=0)
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
        return get_scale_as_float(self.optimizer.get_grad_norm())

    def export_weights(self):
        if self.config.training.build_from_mbridge:
            yield from self.bridge.export_weights(self.model)
        else:
            yield from self.bridge.export_hf_weights(self.model, show_progress=False)

    @override
    def compute_log_probs(
        self,
        rollout_batches: List[Dict[str, List[Any]]],
        compute_pre_logps: bool = True,
        policy_compute_topk: bool = False,
        policy_gather_ids_key: str = None,
        ref_gather_ids_key: str = None,
        skip_ref: bool = False,  # (rionawang)TODO union
    ):
        """Compute log-probs for the policy model and (optionally) the ref model.

        Args:
            compute_pre_logps: Whether to run the policy model forward.
            policy_compute_topk: If True, policy produces its own top-K ids and logprobs.
            policy_gather_ids_key: If set, policy gathers logprobs at batch[key] ids.
                Can be used together with policy_compute_topk.
            ref_gather_ids_key: If set, ref gathers logprobs at batch[key] ids.
                When policy_compute_topk=True and this is set, policy runs first
                (to produce topk_ids that ref depends on).
            skip_ref: If True, skip ref model computation entirely.

        Returns:
            (ref_logps_list, prev_logps_list) — each is a nested list or None.
            When top-K modes are active, elements are dicts with 3D tensor fields.
        """
        assert not self.is_critic_model

        log_prob_top_k = getattr(self.ppo_config, "log_prob_top_k", 0)
        return_per_token_entropy = (
            getattr(self.ppo_config, "loss_func", None) == "steer" or
            getattr(self.ppo_config, "post_compute_logprobs", "none") != "none"
        )

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

        has_ref = not self.policy_config.without_ref and not skip_ref
        # Policy runs first when it produces topk_ids that ref depends on.
        policy_first = (policy_compute_topk and ref_gather_ids_key is not None and has_ref)

        ref_logps = None
        prev_logps = None

        if policy_first:
            # Policy forward first — produce topk_ids, then ref gathers on them.
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
                    batches_list,
                    batch_log_str=batch_log_str.format(name="policy_logprobs"),
                    **policy_kw,
                )
            # Write policy topk_ids into batch for ref to use.
            for i, d in enumerate(prev_logps):
                batches_list[i][ref_gather_ids_key] = d["topk_ids"]

            self.offload_model()
            self.onload_ref_model()
            for model_module in self.ref_model:
                model_module.eval()
            ref_logps = call_logps_func(
                self.ref_model,
                batches_list,
                batch_log_str=batch_log_str.format(name="ref_policy_logprobs"),
                gather_target_ids_key=ref_gather_ids_key,
            )
            self.offload_ref_model()
            self.onload_model()
        else:
            # Ref first (can be offloaded early), then policy.
            if has_ref:
                self.offload_model()
                self.onload_ref_model()
                for model_module in self.ref_model:
                    model_module.eval()
                ref_logps = call_logps_func(
                    self.ref_model,
                    batches_list,
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
                        batches_list,
                        batch_log_str=batch_log_str.format(name="policy_logprobs"),
                        **policy_kw,
                    )
            else:
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

        prev_per_token_entropies_list = None
        if return_per_token_entropy:
            assert prev_logps is not None
            prev_per_token_entropies = [result["prev_per_token_entropy"] for result in prev_logps]
            prev_logps = [result["logprobs"] for result in prev_logps]

        prev_logps_list = None
        if compute_pre_logps:
            prev_logps_list = restor_shape(prev_logps)
            assert prev_logps_list is not None
            if return_per_token_entropy:
                prev_per_token_entropies_list = restor_shape(prev_per_token_entropies)
                assert prev_per_token_entropies_list is not None
        output = (ref_logps_list, prev_logps_list)
        if return_per_token_entropy:
            output += (prev_per_token_entropies_list, )
        return output

    def compute_log_probs_dynamic_cp(
        self,
        rollout_batches: List[Dict[str, List[Any]]],
        compute_pre_logps=True,
    ):
        """Compute log-probs using dynamic CP (THD + Ring Attention path).

        Uses the same forward path as training to ensure numerical consistency
        between prev/ref log-probs and curr_log_probs. Reroute is performed
        once and shared across ref/prev model forward passes.

        Goes through Megatron's ``forward_backward_func`` for PP support and
        progress logging consistency with the non-dynamic-CP path.

        Returns
        -------
        ref_logps_list : list[list[Tensor]] or None
        prev_logps_list : list[list[Tensor]] or None
            Same format as ``compute_log_probs``.
        """
        assert not self.is_critic_model

        samples_per_batch = len(rollout_batches[0]['tokens'])
        batches_list = expand_rollout_batches(rollout_batches)
        num_local_samples = len(batches_list)

        packed_batches, num_micro_batches, _, _, routing_info = (
            self.prepare_data.rl_reroute_data_for_dynamic_cp(
                batches_list,
                self.tokenizer.pad_token_id,
                vocab_size=self.vocab_size,
                pad_with_random_token=self.config.training.moe_pad_with_random_tokens,
                max_seqlen_per_dp_cp_rank=self.dist_config.max_seqlen_per_dp_cp_rank_fwd_only,
            )
        )

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

    @torch.no_grad()
    def _forward_packed_batches_unified(
        self,
        model,
        packed_batches: List[Dict[str, torch.Tensor]],
        num_micro_batches: int,
        seqlen: int,
        batch_log_str: str = "",
    ) -> Dict[int, torch.Tensor]:
        """Forward packed microbatches through forward_backward_func and collect
        per-sample logprobs keyed by global ID.

        Uses the same Megatron pipeline scheduler as the non-dynamic-CP path,
        giving PP support and progress logging for free.
        """
        self.batch_iters = 0
        self.total_iters = num_micro_batches
        self.batch_log_str = batch_log_str

        batch_iter = iter(packed_batches[:num_micro_batches])

        fwd_bwd_function = get_forward_backward_func()
        fwd_results = fwd_bwd_function(
            forward_step_func=self.get_logprob_output_only_func_dynamic_cp(seqlen),
            data_iterator=batch_iter,
            model=model,
            num_microbatches=num_micro_batches,
            forward_only=True,
            seq_length=seqlen,
            micro_batch_size=1,
            collect_non_loss_data=True,
            decoder_seq_length=seqlen,
        )

        # Collect per-sample logprobs keyed by global ID.
        # fwd_results is a list of per-microbatch results (List[torch.Tensor] each).
        # Only populated on the last PP stage.
        per_sample_logprobs: Dict[int, torch.Tensor] = {}
        if mpu.is_pipeline_last_stage():
            for mb_idx, sample_logprobs_list in enumerate(fwd_results):
                sample_ids = packed_batches[mb_idx]["_dyn_cp_sample_ids"]
                if isinstance(sample_ids, torch.Tensor) and not sample_ids.is_cuda:
                    sample_ids = sample_ids.cuda(non_blocking=True)
                assert len(sample_logprobs_list) == sample_ids.shape[0], (
                    f"Expected {sample_ids.shape[0]} samples, got {len(sample_logprobs_list)}"
                )
                for idx, lp in enumerate(sample_logprobs_list):
                    gid = int(sample_ids[idx].item())
                    per_sample_logprobs[gid] = lp

        return per_sample_logprobs

    def _reverse_and_collect(
        self,
        per_sample_logprobs: Dict[int, torch.Tensor],
        global_ids_this_rank: torch.Tensor,
        global_id_logprob_lens,
        gid_to_compute_rank: Dict[int, int],
        gid_to_orig_dcp_rank: Dict[int, List[int]],
        dp_cp_group,
        num_local_samples: int,
    ) -> List[torch.Tensor]:
        """Reverse all-to-all and collect results as CPU tensors.

        Only the last PP stage runs the DP×CP reverse reroute; the per-sample
        list is then broadcast within PP, matching the non-dynamic-CP logprob path.
        """
        if mpu.is_pipeline_last_stage():
            restored_logprobs = reverse_reroute_logprobs(
                per_sample_logprobs,
                global_ids_this_rank,
                global_id_logprob_lens,
                gid_to_compute_rank,
                gid_to_orig_dcp_rank,
                dp_cp_group,
            )

            result = []
            for i in range(num_local_samples):
                gid = int(global_ids_this_rank[i])
                lp = restored_logprobs[gid]
                result.append(lp.float().cpu())
            clear_memory()
        else:
            result = []

        return BroadcastUtils.broadcast_object_within_pp(result)

    def _rl_collect_mb_dumped_metrics(self, _metric, dumped_metrics_per_ppo_step):
        """each microbatch collect dumped loss_fn and moe metrics"""
        dumped_loss_fn_metrics = _metric.pop("dumped_loss_fn_metrics", None)
        if dumped_metrics_per_ppo_step is None or dumped_loss_fn_metrics is None:
            return
        dump_moe_topk = getattr(self.training_config, "ppo_dump_moe_topk", 0) or 0
        if dump_moe_topk > 0 and not self.dist_config.dynamic_context_parallel:
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

            mb_num_microbatches = num_microbatches
            (
                self._step_global_batch_size,
                self._step_effective_global_batch_size,
                self._step_global_token_cnt,
            ) = self._compute_step_gbs_and_token_cnt(batch)
            dyn_cp_stats = None
            routing_info = None
            if self.config.policy.dist_config.dynamic_context_parallel:
                batch, mb_num_microbatches, seqlen_sum, seqlen_sq_sum, routing_info = (
                    self.prepare_data.rl_reroute_data_for_dynamic_cp(
                        batch,
                        self.tokenizer.pad_token_id,
                        vocab_size=self.vocab_size,
                        pad_with_random_token=self.config.training.moe_pad_with_random_tokens,
                        need_routing_info=self.should_dump_metrics,
                    )
                )
                dyn_cp_stats = {
                    "policy/dyn_cp_seqlen_sum": seqlen_sum,
                    "policy/dyn_cp_seqlen_sq_sum": seqlen_sq_sum,
                    "policy/dyn_cp_num_micro_batches": mb_num_microbatches,
                }

            # maybe enable r3 replay if required
            with self.get_router_replay_ctx():
                _metric = self._update_policy(
                    batch,
                    num_microbatches=mb_num_microbatches,
                    routing_info=routing_info,
                )

            if dyn_cp_stats is not None:
                _metric.update(dyn_cp_stats)

            self._rl_collect_mb_dumped_metrics(_metric, dumped_metrics_per_ppo_step)

            if clear_gathered_routing_info is not None:
                clear_gathered_routing_info()

            extend_value_to_dict(
                metrics, _metric
            )  # append the metric from this micro-batch to global metrics.

            update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()
            extend_value_to_dict(metrics, {"policy/grad_norm": get_scale_as_float(grad_norm)})
            post_clip_grad_norm = self.maybe_post_clip_grad_norm(update_successful)
            if post_clip_grad_norm is not None:
                extend_value_to_dict(metrics, {"policy/post_clip_grad_norm": post_clip_grad_norm})

            if update_successful:
                if self.config.optimizer.update_lr_by_train_step:
                    self.optimizer_scheduler.step(1)
                if self.ppo_config.use_adaptive_entropy:
                    assert "policy/scaled_entropy" in _metric, (
                        "use_adaptive_entropy requires policy/scaled_entropy in train metrics"
                    )
                    extend_value_to_dict(
                        metrics,
                        update_adaptive_entropy_after_train_step(
                            self.ppo_config,
                            float(_metric["policy/scaled_entropy"]),
                        ),
                    )
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
        grad_norm = reduce_max_stat_across_model_parallel_group(get_scale_as_float(grad_norm))
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

    @override
    def pretrain_step(self, batch: List[Dict[str, Any]], num_microbatches: int, step: int):
        """Packed-THD pretrain step. Microbatches may stay on CPU until each forward.

        Uses ``prepare_data.pretrain_packed`` (not ``sft_train`` / expand).
        Expects dataset-packed flat fields (``tokens``, ``cu_seqlens_padded``, ...).
        """
        assert num_microbatches == len(batch)
        assert num_microbatches >= 1
        for model_chunk in self.model:
            model_chunk.zero_grad_buffer()
        self.optimizer.zero_grad()

        num_samples, num_tokens, num_label_tokens, num_pad_tokens = (
            self._compute_pretrain_packed_batch_stats(batch)
        )
        metric = self._pretrain_step(batch, num_microbatches, forward_only=False)
        update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()
        post_clip_grad_norm = self.maybe_post_clip_grad_norm(update_successful)
        lr = self.step_and_get_lr()

        (
            update_successful,
            grad_norm,
            num_zeros_in_grad,
            post_clip_grad_norm,
        ) = self.reduce_optimizer_stats_across_model_parallel_group(
            update_successful,
            get_scale_as_float(grad_norm),
            get_scale_as_float(num_zeros_in_grad),
            get_scale_as_float(post_clip_grad_norm),
        )

        metric["pretrain/num_samples_sum"] = num_samples
        metric["pretrain/num_tokens_sum"] = num_tokens
        metric["pretrain/num_label_tokens_sum"] = num_label_tokens
        metric["pretrain/num_pad_tokens_sum"] = num_pad_tokens
        metric["pretrain/grad_norm"] = grad_norm
        metric["pretrain/lr"] = lr
        metric["pretrain/num_zeros_in_grad"] = num_zeros_in_grad
        if post_clip_grad_norm is not None:
            metric["pretrain/post_clip_grad_norm"] = post_clip_grad_norm

        if self.policy_config.manual_clear_memory and (
            step + 1
        ) % self.policy_config.manual_clear_memory_interval == 0:
            clear_memory()
        return metric

    @torch.no_grad()
    def eval_step(self, batch: List[Dict[str, Any]], num_microbatches: int):
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
        metric = self._finetune_step(batch, num_microbatches=num_microbatches, forward_only=True)
        if self.config.policy.dist_config.dynamic_context_parallel:
            metric["eval/dyn_cp_seqlen_sum"] = seqlen_sum
            metric["eval/dyn_cp_seqlen_sq_sum"] = seqlen_sq_sum
            metric["eval/dyn_cp_num_micro_batches"] = num_microbatches
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
            return self._compute_logits_or_hidden_states(rollout_batches, return_logits=True)

    def compute_hidden_states(
        self, rollout_batches: List[Dict[str, torch.Tensor]]
    ) -> tuple[None, List[Optional[torch.Tensor]]]:
        return self._compute_logits_or_hidden_states(rollout_batches, return_logits=False)

    def _compute_logits_or_hidden_states(
        self,
        rollout_batches: List[Dict[str, torch.Tensor]],
        *,
        return_logits: bool,
    ) -> tuple[None, List[Optional[torch.Tensor]]]:
        begine_t = sync_cuda_and_get_time()
        assert rollout_batches[0]['tokens'].ndim == 1, f"{rollout_batches[0]['tokens'].ndim}"

        assert self.policy_config.without_ref

        self.set_model_eval()
        output = self._compute_logits_or_hidden_states_impl(
            self.model,
            rollout_batches,
            batch_log_str=(
                "get_policy_logits microbatch "
                if return_logits else "get_teacher_hidden_states microbatch "
            ),
            return_logits=return_logits,
        )
        end_t = sync_cuda_and_get_time()

        if mpu.is_pipeline_last_stage():
            assert output is not None
            output_name = "logits" if return_logits else "hidden_states"
            log(
                f"compute_{output_name} using time {end_t - begine_t}",
                rank=0,
            )
        else:
            output = [None for _ in range(len(rollout_batches))]
        return None, output

    def default_compute_logits(self, rollout_batches: List[Dict[str, torch.Tensor]]):
        return self._compute_logits_or_hidden_states(rollout_batches, return_logits=True)

    @override
    def set_model_eval(self):
        for model_module in self.model:
            model_module.eval()

    @override
    def set_model_train(self):
        for model_module in self.model:
            model_module.train()

    def _align_values_to_rollout_logprobs(
        self,
        rollout_batches: List[Dict[str, List[Any]]],
        values_list: List[List[torch.Tensor]],
    ):
        """Project critic values onto each sample's policy-logprob coordinates."""
        for rollout_batch, batch_values in zip(rollout_batches, values_list, strict=True):
            logprobs = rollout_batch.get("logprobs")
            if logprobs is None:
                continue
            prompt_lengths = rollout_batch.get("prompt_lengths")
            sequence_lengths = rollout_batch.get("sequence_lengths")

            def to_int(x):
                return int(x.item()) if hasattr(x, "item") else int(x)

            for i, (value, logprob) in enumerate(zip(batch_values, logprobs, strict=True)):
                target_len = logprob.size(-1)
                value_len = value.size(-1)
                if value_len == target_len:
                    continue
                if value_len > target_len:
                    batch_values[i] = value[:target_len].contiguous()
                    continue

                prefix_len = None
                response_len = None
                if prompt_lengths is not None:
                    prefix_len = max(to_int(prompt_lengths[i]) - 1, 0)
                if prompt_lengths is not None and sequence_lengths is not None:
                    response_len = max(
                        to_int(sequence_lengths[i]) - to_int(prompt_lengths[i]),
                        0,
                    )

                padded_prefix_len = target_len - value_len
                if (
                    prefix_len is not None and padded_prefix_len > 0 and
                    padded_prefix_len <= prefix_len and
                    (response_len is None or value_len >= response_len)
                ):
                    batch_values[i] = torch.nn.functional.pad(
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
                    batch_values[i] = torch.nn.functional.pad(
                        value,
                        (prefix_len, right_pad),
                        value=0.0,
                    ).contiguous()
                    continue

                raise RuntimeError(
                    "critic values shorter than policy logprobs after CP gather: "
                    f"sample={i}, values_shape={value.shape}, logprobs_shape={logprob.shape}, "
                    f"prompt_length={None if prompt_lengths is None else prompt_lengths[i]}, "
                    f"sequence_length={None if sequence_lengths is None else sequence_lengths[i]}, "
                    f"padded_prefix_len={padded_prefix_len}"
                )
        return values_list

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
        return self._align_values_to_rollout_logprobs(rollout_batches, values_list)

    @torch.no_grad()
    def _smart_pad_value_forward_step(
        self,
        batch_iter,
        num_microbatches,
        micro_batch_size,
        seq_length,
    ):
        fwd_bwd_function = get_forward_backward_func()
        value_microbatches = fwd_bwd_function(
            forward_step_func=self.get_logits_or_hidden_state_only_func(
                seq_length, inference_only=True
            ),
            data_iterator=batch_iter,
            model=self._smart_pad_current_model,
            num_microbatches=num_microbatches,
            forward_only=True,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            collect_non_loss_data=True,
            decoder_seq_length=seq_length,
        )
        if len(value_microbatches) == 0:
            clear_memory()
            return value_microbatches

        values = torch.cat(value_microbatches).squeeze(-1)
        if mpu.get_context_parallel_world_size() > 1:
            values = all_gather_from_context_parallel_region(values)
        values = values[:, :-1].clone()

        ret = []
        offset = 0
        for value_microbatch in value_microbatches:
            cur_mbs = value_microbatch.size(0)
            ret.append(values[offset:offset + cur_mbs])
            offset += cur_mbs
        clear_memory()
        return ret

    @torch.no_grad()
    def smart_pad_compute_values(self, rollout_batches: List[Dict[str, List[Any]]]):
        samples_per_batch = len(rollout_batches[0]['tokens'])
        batches_list = expand_rollout_batches(rollout_batches)
        assert len(batches_list) == samples_per_batch * len(rollout_batches)

        self.onload_model()
        for model_module in self.model:
            model_module.eval()

        self.batch_iters = 0
        total_samples = len(batches_list)
        self.total_iters = total_samples // self.forward_only_mbs
        self.batch_log_str = "[smart_pad] get_values microbatch "
        self._smart_pad_current_model = self.model

        dynamic_mbs_target_seqlen = getattr(
            self.policy_config, 'dynamic_mbs_target_seqlen_fwd_only', None
        )
        dynamic_mbs_limit = getattr(self.policy_config, 'dynamic_mbs_limit_fwd_only', None)

        smart_pad_helper = CatedSmartPadInferHelper(batches_list, self.forward_only_mbs)
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
            values_list = smart_pad_helper.get_rowed_based_forward_results(is_row_based_rets=True)
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

        grouped_values = []
        bs = len(values) // samples_per_batch
        assert bs * samples_per_batch == len(values)
        for i in range(bs):
            grouped_values.append(values[i * samples_per_batch:(i + 1) * samples_per_batch])
        clear_memory()
        return self._align_values_to_rollout_logprobs(rollout_batches, grouped_values)

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
            extend_value_to_dict(metrics, {"value/grad_norm": get_scale_as_float(grad_norm)})
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
