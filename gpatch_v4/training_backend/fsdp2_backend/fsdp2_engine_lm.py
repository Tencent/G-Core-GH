from contextlib import nullcontext
from typing import Any, Dict, List

import torch
from torch.distributed.tensor import DTensor
from transformers import AutoConfig, AutoTokenizer
from typing_extensions import override

from megatron.core import mpu
from megatron.core.utils import divide

from gpatch_v4.configs.config import OnPolicyDistillConfig
from gpatch_v4.core.seqlen_balancing import convert_mbs_for_pack_seq
from gpatch_v4.extended_model import (
    CheckpointContextFnFactory,
    PostInitModelFactory,
    PrepareDataForwardFactory,
    ResetRouterCorrectionBiasAccumFactory,
    UpdateRouterCorrectionBiasFactory,
)
from gpatch_v4.training_backend.base_engine import BaseEngine
from gpatch_v4.training_backend.common.swap_mixin import EngineSwapMixin
from gpatch_v4.training_backend.fsdp2_backend.checkpoint import (
    get_latest_checkpoint_folder,
    load_checkpoint,
)
from gpatch_v4.training_backend.fsdp2_backend.fsdp2_swap_impl import Fsdp2SwapImpl
from gpatch_v4.training_backend.fsdp2_backend.linear_ce import (
    install_linear_ce_head_bypass,
)
from gpatch_v4.training_backend.fsdp2_backend.mixin import (
    CheckpointMixin,
    ForwardStepMixin,
    Fsdp2EngineMixin,
)
from gpatch_v4.training_backend.fsdp2_backend.optimizer import (
    setup_lr_scheduler,
    setup_optimizer,
)
from gpatch_v4.training_backend.fsdp2_backend.weight_exportor import get_weight_exportor
from gpatch_v4.training_backend.loss_factory import (
    PolicyLossInput,
    get_policy_loss_fn,
    is_seq_mean_rl_loss_fn,
)
from gpatch_v4.utils import (
    cache_hf_metadata_files,
    clear_memory,
    expand_rollout_batches,
    extend_value_to_dict,
    get_batches_max_seqlen,
    get_k_split_list,
    get_max_seqlen_within_dp,
    log,
    logging_memory_usage,
    logging_rank0,
    logical_and_across_model_parallel_group,
    masked_mean,
    profile_memory_and_time,
    reduce_max_stat_across_model_parallel_group,
    sync_cuda_and_get_time,
)
from gpatch_v4.utils.grpo_thd_alignment import build_grpo_thd_dump_records


class Fsdp2EngineLm(
    BaseEngine, EngineSwapMixin, Fsdp2EngineMixin, ForwardStepMixin, CheckpointMixin
):
    def __init__(self, config, policy_config, tokenizer: AutoTokenizer):
        super().__init__(config, policy_config, tokenizer)

        self._setup_device_mesh()
        self.swap_impl = Fsdp2SwapImpl()
        self.prepare_data = PrepareDataForwardFactory.get_prepare_data_fwd(config)
        self.post_init_model = PostInitModelFactory.get_post_init_model(config)
        self.checkpoint_context_fn = CheckpointContextFnFactory.get_checkpoint_context_fn(config)
        self.reset_router_correction_bias_accum = ResetRouterCorrectionBiasAccumFactory.get_reset_router_correction_bias_accum(
            config
        )
        self.update_router_correction_bias = UpdateRouterCorrectionBiasFactory.get_update_router_correction_bias(
            config
        )

        cache_hf_metadata_files(
            self.policy_config.hf_model_path,
            self.checkpoint_config.save_ckpt_path,
        )
        self.hf_config = AutoConfig.from_pretrained(self.policy_config.hf_model_path)
        self.forward_only_mbs = self.policy_config.forward_only_mbs

    def setup_ref_model(self, init_context):
        ref_hf_model_path = self.policy_config.ref_hf_model_path
        load_latest_step = None  #get_latest_checkpoint_folder(self.checkpoint_config.load_ref_ckpt_path)
        if load_latest_step is not None:
            ref_hf_model_path = load_latest_step
        log(f"creating ref model from {ref_hf_model_path}", rank=0)
        self.ref_model = self.get_fsdp2_model(init_context, ref_hf_model_path)
        if self.training_config.use_linear_ce:
            install_linear_ce_head_bypass(self.ref_model)
        with profile_memory_and_time(f"offload_ref_model", rank=0):
            self.offload_ref_model()

    @override
    def setup_model_and_get_optimizer(self):
        init_context = self._get_init_weight_context_manager()

        if not self.policy_config.without_ref:
            self.setup_ref_model(init_context)
        else:
            self.ref_model = None
            self.get_swap_state().ref_model = False

        log(f"creating model from {self.policy_config.hf_model_path}", rank=0)
        self.model = self.get_fsdp2_model(init_context, self.policy_config.hf_model_path)

        if self.config.training.recompute:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": False,
                    "context_fn": self.checkpoint_context_fn,
                }
            )

        self.post_init_model(self.model)
        if self.training_config.use_linear_ce:
            install_linear_ce_head_bypass(self.model)

        if self.policy_config.without_optim:
            self.optimizer = None
            self.lr_scheduler = None
            self.hf_config = self.model.config
            return 0

        if self.config.debug.debug_no_optim:
            optimizer = None
        else:
            optimizer = setup_optimizer(self.config, self.model, latest_step=None)
        self.optimizer = optimizer

        self.lr_scheduler = setup_lr_scheduler(self.config, self.optimizer, latest_step=None)
        self.hf_config = self.model.config
        latest_saved_step = 0

        load_latest_step = get_latest_checkpoint_folder(self.checkpoint_config.load_ckpt_path)
        if load_latest_step is not None:
            latest_saved_step = load_checkpoint(
                self.config, self.model, self.optimizer, self.lr_scheduler, load_latest_step
            )
        return latest_saved_step

    def export_weights(self):
        """Yield ``(name, tensor)`` for weight synchronization.

        Arch-specific exporters (see :mod:`weight_exportor`) may rename,
        gather EP shards, and quantize — e.g. DSV4 disk keys for vLLM.
        The default path calls ``full_tensor()`` on each DTensor parameter.
        Tensors stay on GPU for CUDA IPC across Ray actors.
        """
        weight_exportor = get_weight_exportor(self.config.policy.model_arch)
        if weight_exportor is not None:
            yield from weight_exportor(self.model)
            return

        for name, param in self.model.named_parameters():
            assert isinstance(param, DTensor)
            yield name, param.full_tensor().detach()

    @override
    def compute_log_probs(
        self,
        rollout_batches: List[Dict[str, List[Any]]],
        compute_pre_logps=True,
    ):
        return self.normal_compute_log_probs(rollout_batches, compute_pre_logps)

    @override
    def rl_train_actor(self, dataloader_iter):
        pack_seq = self.policy_config.ppo_pack_seq
        if pack_seq:
            assert self.policy_config.model_arch == "deepseek_v4", (
                "FSDP2 GRPO THD currently supports only deepseek_v4"
            )
        if isinstance(self.config, OnPolicyDistillConfig):
            assert not pack_seq, "FSDP2 OPD + DSV4 THD is not supported yet"
        self.set_model_train()

        loss_fn = get_policy_loss_fn(self.config.ppo.loss_func)
        seq_mean_loss = is_seq_mean_rl_loss_fn(loss_fn)
        calculate_per_token_loss = self.policy_config.override_transformer_config.get(
            "calculate_per_token_loss", False
        )
        metrics = {}
        dumped_metrics_per_ppo_step = (
            [] if self.should_dump_metrics and pack_seq and mpu.get_context_parallel_rank() == 0
            else None
        )

        for batch in dataloader_iter:
            self.optimizer.zero_grad()
            dumped_batch = [None] * len(batch) if dumped_metrics_per_ppo_step is not None else None
            sample_order = (
                {
                    id(sample): idx
                    for idx, sample in enumerate(batch)
                } if dumped_batch is not None else None
            )
            if not self.training_config.freeze_router_correction_bias:
                self.reset_router_correction_bias_accum(self.model)
            global_token_count = None
            if seq_mean_loss and calculate_per_token_loss:
                local_token_count = 0.0
                for sample in batch:
                    sample_count = float(torch.as_tensor(sample["mask"]).sum().item())
                    local_token_count += sample_count
                global_token_count = torch.tensor(
                    local_token_count, dtype=torch.float32, device="cuda"
                )
                torch.distributed.all_reduce(global_token_count, group=self.dp_group)
                assert global_token_count.item(
                ) > 0, ("per-token RL loss requires at least one active response token")
            if pack_seq:
                seq_length = self.training_config.seq_length
                data_iter = convert_mbs_for_pack_seq(
                    batch,
                    max_token_len=seq_length,
                    pad_each_doc_to_multi_of=self.prepare_data._pad_each_doc_to_multi_of,
                    dp_group=mpu.get_data_parallel_group(),
                    cp_size=mpu.get_context_parallel_world_size(),
                )
                num_microbatches = len(data_iter)
            else:
                num_microbatches = divide(
                    self.training_config.train_gbs,
                    self.training_config.train_mbs * self.dp_size,
                )
                seq_length = get_batches_max_seqlen(batch, self.training_config.pad_to_mulitiple_of)
                seq_length = get_max_seqlen_within_dp(seq_length)
                data_iter = get_k_split_list(batch, num_microbatches)

            use_r3 = self.config.training.moe_router_replay

            if isinstance(self.config, OnPolicyDistillConfig):
                prepare_data_func = self.prepare_data.opd_train
            else:
                prepare_data_func = self.prepare_data.grpo_train

            for batches in data_iter:
                batch_data, fwd_kwargs = prepare_data_func(
                    batches,
                    seq_length,
                    self.tokenizer.pad_token_id,
                    ppo_pack_seq=pack_seq,
                    pad_with_random_token=self.config.training.moe_pad_with_random_tokens,
                    vocab_size=self._get_vocab_size(),
                )
                full_psp = batch_data.get("full_packed_seq_params")
                if full_psp is not None and not getattr(self, "_logged_grpo_thd", False):
                    log(
                        f"[FSDP2 GRPO THD] qkv_format={full_psp.qkv_format}, "
                        f"segments={full_psp.cu_seqlens_q.numel() - 1}, "
                        f"packed_tokens={full_psp.total_seqlen}, "
                        f"local_tokens={fwd_kwargs['input_ids'].shape[1]}, "
                        f"cp_size={mpu.get_context_parallel_world_size()}",
                        rank=0,
                    )
                    self._logged_grpo_thd = True

                replay_ctx = (
                    self._maybe_router_replay(
                        self.model,
                        batches,
                        seq_length,
                        packed_seq_params=full_psp,
                    ) if use_r3 else nullcontext()
                )
                with replay_ctx:
                    train_forward_context = (
                        nullcontext()
                        if self.training_config.recompute else self.checkpoint_context_fn()[0]
                    )
                    target = batch_data["target"]
                    pre_shifted = full_psp is not None
                    response_mask = batch_data["mask"]

                    # Keep linear_ce context through backward (recompute-safe).
                    with self._maybe_rl_linear_ce_context(
                        self.model,
                        target,
                        pre_shifted=pre_shifted,
                        return_entropy=True,
                    ):
                        with train_forward_context:
                            model_out = self.model(**fwd_kwargs)
                        curr_log_probs, entropy = self._rl_logprobs_and_entropy_from_head_output(
                            model_out.logits,
                            target,
                            response_mask,
                            pre_shifted=pre_shifted,
                            allow_compile=True,
                        )

                        assert curr_log_probs.shape == response_mask.shape, (
                            f"current logprob shape {tuple(curr_log_probs.shape)} "
                            f"must match response mask shape {tuple(response_mask.shape)}"
                        )
                        if dumped_batch is not None:
                            assert sample_order is not None
                            dump_records = build_grpo_thd_dump_records(
                                batches,
                                batch_data,
                                curr_log_probs,
                            )
                            for sample, record in zip(batches, dump_records, strict=True):
                                dumped_batch[sample_order[id(sample)]] = record
                        scaled_entropy = masked_mean(entropy, response_mask)

                        advantages = batch_data["advantages"]

                        loss_input = PolicyLossInput(
                            advantages=advantages,
                            prev_log_probs=batch_data["prev_log_probs"],
                            ref_log_probs=batch_data.get("ref_log_probs", None),
                            curr_log_probs=curr_log_probs,
                            response_mask=response_mask,
                            scaled_entropy=scaled_entropy,
                            rollout_log_probs=batch_data.get("rollout_log_probs", None),
                            per_token_entropy=entropy,
                            teacher_log_probs=batch_data.get("teacher_log_probs", None),
                            entropy_aux_figures=batch_data.get("entropy_aux_figures", None),
                            cu_seqlens_padded=batch_data.get("cu_seqlens_padded", None),
                            local_cp_size=1,
                            calculate_per_token_loss=calculate_per_token_loss,
                            should_dump_metrics=False,
                        )

                        bwd_loss, step_metrics = loss_fn(self.config, loss_input)
                        # Keep router_replay_ctx alive through backward: gradient
                        # checkpointing re-runs forward during recompute and still
                        # needs the pinned expert indices from rollout.
                        if seq_mean_loss:
                            if calculate_per_token_loss:
                                # bwd_loss is this micro-batch's token-sum numerator.
                                # FSDP averages DP gradients, so multiply by dp_size
                                # and divide once by the global active-token count.
                                (bwd_loss * self.dp_size / global_token_count).backward()
                            else:
                                # bwd_loss is the sum of per-sequence token means.
                                (bwd_loss * self.dp_size /
                                 self.training_config.train_gbs).backward()
                        else:
                            (bwd_loss / num_microbatches).backward()
                extend_value_to_dict(metrics, {f"policy/{k}": v for k, v in step_metrics.items()})

            if dumped_batch is not None:
                assert all(record is not None for record in dumped_batch)
                dumped_metrics_per_ppo_step.extend(dumped_batch)

            grad_norm = self.clip_grad_norm_()
            if not torch.isfinite(grad_norm):
                log(f"WARN: grad_norm is not finite: {grad_norm}")
                self.optimizer.zero_grad()
            else:
                self.optimizer.step()
            if not self.training_config.freeze_router_correction_bias:
                maxvio_max, maxvio_mean = self.update_router_correction_bias(
                    model=self.model,
                    update_speed=self.training_config.router_correction_bias_update_speed,
                    use_abs_update=self.training_config.router_correction_bias_use_abs_update,
                )
                extend_value_to_dict(metrics, {"policy/maxvio_max": maxvio_max})
                extend_value_to_dict(metrics, {"policy/maxvio_mean": maxvio_mean})
            extend_value_to_dict(metrics, {"policy/grad_norm": grad_norm})
            extend_value_to_dict(metrics, {"policy/seq_length": seq_length})

        clear_memory()
        reduced_metrics = {}
        for key, val in metrics.items():
            if isinstance(val, list):
                if len(val) == 0:
                    continue
                if all(isinstance(x, torch.Tensor) for x in val):
                    stacked = torch.stack([x.detach() for x in val])
                    if stacked.dim() == 2 and stacked.shape[1] == 2:
                        # Token-level metrics are represented as [sum, count].
                        # Aggregate across local micro-batches, then across DP ranks.
                        sum_and_count = stacked.sum(dim=0)
                        torch.distributed.all_reduce(sum_and_count, group=self.dp_group)
                        reduced_metrics[key] = (sum_and_count[0] /
                                                sum_and_count[1].clamp(min=1)).cpu().item()
                    else:
                        reduced_metrics[key] = stacked.float().mean().cpu().item()
                else:
                    scalar_values = [
                        x.detach().cpu().item() if isinstance(x, torch.Tensor) else x for x in val
                    ]
                    reduced_metrics[key] = sum(scalar_values) / len(scalar_values)
            elif isinstance(val, torch.Tensor):
                reduced_metrics[key] = val.detach().cpu().item()
            else:
                reduced_metrics[key] = val
        if dumped_metrics_per_ppo_step is not None:
            reduced_metrics["dumped_metrics_per_ppo_step"] = dumped_metrics_per_ppo_step
        return reduced_metrics

    @override
    def finetune_step(self, batch: List[Dict[str, Any]], num_microbatches: int, step: int):
        self.set_model_train()
        self.optimizer.zero_grad()
        if not self.training_config.freeze_router_correction_bias:
            self.reset_router_correction_bias_accum(self.model)

        metric = self._finetune_step(batch, num_microbatches=num_microbatches)

        grad_norm = self.clip_grad_norm_()
        if not torch.isfinite(grad_norm):
            log(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
        if not self.training_config.freeze_router_correction_bias:
            maxvio_max, maxvio_mean = self.update_router_correction_bias(
                model=self.model,
                update_speed=self.training_config.router_correction_bias_update_speed,
                use_abs_update=self.training_config.router_correction_bias_use_abs_update,
            )
            metric["finetune/maxvio_max"] = maxvio_max
            metric["finetune/maxvio_mean"] = maxvio_mean
        lr = self.step_and_get_lr()

        metric["finetune/grad_norm"] = grad_norm
        metric["finetune/lr"] = lr
        return metric

    @override
    def pretrain_step(self, batch: List[Dict[str, Any]], num_microbatches: int, step: int):
        raise NotImplementedError("pretrain_step is not implemented")

    @override
    def set_model_eval(self, model=None):
        if model is not None:
            model.eval()
        else:
            self.model.eval()

    @override
    def set_model_train(self, model=None):
        if model is not None:
            model.train()
        else:
            self.model.train()

    def step_and_get_lr(self):
        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_last_lr()[0]
        return lr

    def normal_compute_log_probs(
        self,
        rollout_batches: List[Dict[str, List[Any]]],
        compute_pre_logps=True,
    ):
        samples_per_batch = len(rollout_batches[0]['tokens'])
        batches_list = expand_rollout_batches(rollout_batches)
        assert len(batches_list) == samples_per_batch * len(rollout_batches)
        ref_logps = None
        prev_logps = None

        if not self.policy_config.without_ref:
            self.offload_model()
            self.onload_ref_model()
            self.set_model_eval(self.ref_model)
            ref_logps = self.compute_logprobs(
                self.ref_model,
                batches_list,
                batch_log_str="get_ref_policy_logprobs",
            )
            self.offload_ref_model()

        use_r3 = self.config.training.moe_router_replay
        if compute_pre_logps:
            self.onload_model()
            self.set_model_eval(self.model)
            prev_logps = self.compute_logprobs(
                self.model,
                batches_list,
                batch_log_str="get_policy_logprobs",
                enable_r3=use_r3,
            )

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

        prev_logps_list = restor_shape(prev_logps)
        if compute_pre_logps:
            assert prev_logps_list is not None

        return ref_logps_list, prev_logps_list
