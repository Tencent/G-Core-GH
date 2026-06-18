from typing import Any, Dict, List

import torch
from torch.distributed.tensor import DTensor
from transformers import AutoConfig, AutoTokenizer
from typing_extensions import override

from megatron.core import mpu
from megatron.core.utils import divide

from gpatch_v4.extended_model import PrepareDataForwardFactory
from gpatch_v4.training_backend.base_engine import BaseEngine
from gpatch_v4.training_backend.common.swap_mixin import EngineSwapMixin
from gpatch_v4.training_backend.fsdp2_backend.checkpoint import (
    get_latest_checkpoint_folder,
    load_checkpoint,
)
from gpatch_v4.training_backend.fsdp2_backend.fsdp2_swap_impl import Fsdp2SwapImpl
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
from gpatch_v4.training_backend.loss_factory import PolicyLossInput, get_policy_loss_fn
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


class Fsdp2EngineLm(
    BaseEngine, EngineSwapMixin, Fsdp2EngineMixin, ForwardStepMixin, CheckpointMixin
):
    def __init__(self, config, policy_config, tokenizer: AutoTokenizer):
        super().__init__(config, policy_config, tokenizer)

        self._setup_device_mesh()
        self.swap_impl = Fsdp2SwapImpl()
        self.prepare_data = PrepareDataForwardFactory.get_prepare_data_fwd(config)

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
        self.ref_model = self.get_fsdp2_model(init_context, self.policy_config.hf_model_path)
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
            self.model.gradient_checkpointing_enable()

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
        num_microbatches = divide(
            self.training_config.train_gbs, self.training_config.train_mbs * self.dp_size
        )
        self.set_model_train()

        loss_fn = get_policy_loss_fn(self.config.ppo.advantage_type)
        metrics = {}

        for batch in dataloader_iter:
            self.optimizer.zero_grad()
            seq_length = get_batches_max_seqlen(batch, self.training_config.pad_to_mulitiple_of)
            seq_length = get_max_seqlen_within_dp(seq_length)

            data_iter = get_k_split_list(batch, num_microbatches)

            for batches in data_iter:
                batch_data, fwd_kwargs = self.prepare_data.grpo_train(
                    batches,
                    seq_length,
                    self.tokenizer.pad_token_id,
                    ppo_pack_seq=False,
                )

                logits = self.model(**fwd_kwargs).logits.float()
                target = batch_data["target"]

                # CP-aware logprobs
                curr_log_probs = self.gather_log_probs_packed(logits, target, allow_compile=True)
                response_mask = batch_data["mask"]

                # Entropy: compute on all positions per rank, then all-gather
                probs = logits.softmax(dim=-1)
                entropy = -(probs * logits.log_softmax(dim=-1)).sum(dim=-1)
                entropy = self._all_gather_cp_aware(entropy)
                entropy = entropy[:, :-1]
                scaled_entropy = masked_mean(entropy, response_mask)

                advantages = batch_data["advantages"]

                loss_input = PolicyLossInput(
                    advantages=advantages,
                    prev_log_probs=batch_data["prev_log_probs"],
                    ref_log_probs=batch_data["ref_log_probs"],
                    curr_log_probs=curr_log_probs,
                    response_mask=response_mask,
                    scaled_entropy=scaled_entropy,
                    rollout_log_probs=batch_data.get("rollout_log_probs", None),
                    per_token_entropy=entropy,
                    should_dump_metrics=False,
                )

                bwd_loss, step_metrics = loss_fn(self.config, loss_input)
                (bwd_loss / num_microbatches).backward()
                extend_value_to_dict(metrics, {f"policy/{k}": v for k, v in step_metrics.items()})

            grad_norm = self.clip_grad_norm_()
            if not torch.isfinite(grad_norm):
                log(f"WARN: grad_norm is not finite: {grad_norm}")
                self.optimizer.zero_grad()
            else:
                self.optimizer.step()
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
        return reduced_metrics

    @override
    def finetune_step(self, batch: List[Dict[str, Any]], num_microbatches: int, step: int):
        self.set_model_train()
        self.optimizer.zero_grad()

        metric = self._finetune_step(batch, num_microbatches=num_microbatches)

        grad_norm = self.clip_grad_norm_()
        if not torch.isfinite(grad_norm):
            log(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
        lr = self.step_and_get_lr()

        metric["finetune/grad_norm"] = grad_norm
        metric["finetune/lr"] = lr
        return metric

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

        if compute_pre_logps:
            self.onload_model()
            self.set_model_eval(self.model)
            prev_logps = self.compute_logprobs(
                self.model,
                batches_list,
                batch_log_str="get_policy_logprobs",
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
