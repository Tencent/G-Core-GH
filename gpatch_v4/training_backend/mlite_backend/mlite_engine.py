import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from transformers import AutoConfig, AutoTokenizer
from typing_extensions import override

from megatron.core import mpu
from megatron.lite.runtime import create_runtime
from megatron.lite.runtime.contracts.config import RuntimeConfig

from gpatch_v4.configs import FinetuneConfig
from gpatch_v4.extended_model import PrepareDataForwardFactory
from gpatch_v4.training_backend.base_engine import BaseEngine
from gpatch_v4.training_backend.common.swap_mixin import EngineSwapMixin
from gpatch_v4.training_backend.loss import MliteFinetuneLossInput, get_loss_fn
from gpatch_v4.training_backend.mlite_backend.config import (
    build_mlite_config,
    resolve_model_name,
    validate_mlite_config,
)
from gpatch_v4.training_backend.mlite_backend.lr_scheduler import build_lr_scheduler
from gpatch_v4.training_backend.mlite_backend.mixin import CheckpointMixin, MetricsMixin
from gpatch_v4.training_backend.mlite_backend.mlite_swap_impl import MliteSwapImpl
from gpatch_v4.utils import cache_hf_metadata_files


class MliteEngine(BaseEngine, EngineSwapMixin, CheckpointMixin, MetricsMixin):
    """Finetune engine for the Megatron Lite runtime."""
    def __init__(
        self,
        config: FinetuneConfig,
        policy_config,
        tokenizer: AutoTokenizer,
        is_critic_model: bool = False,
    ):
        if not isinstance(config, FinetuneConfig):
            raise NotImplementedError("mlite currently supports FinetuneConfig only")
        if is_critic_model:
            raise NotImplementedError("mlite finetune does not support critic models")
        super().__init__(config, policy_config, tokenizer)
        self.is_critic_model = False
        validate_mlite_config(self)
        self.hf_config = AutoConfig.from_pretrained(
            self.policy_config.hf_model_path,
            trust_remote_code=True,
        )
        model_name = resolve_model_name(self.hf_config)
        if model_name not in ("qwen3_5", "welm_v4_5"):
            raise NotImplementedError(
                "mlite finetune currently supports Qwen3.5 and WELM4.5 models only"
            )
        if self.checkpoint_config.convert_mcore_to_hf_online:
            cache_hf_metadata_files(
                self.policy_config.hf_model_path,
                self.checkpoint_config.save_ckpt_path,
            )

        self.prepare_data = PrepareDataForwardFactory.get_prepare_data_fwd(config)
        self.runtime = None
        self.handle = None
        self.optimizer = None
        self.optimizer_scheduler = None
        self._mlite_config = None
        self.swap_impl = MliteSwapImpl()
        #TODO: 这里先hard code了，等外主框架合并
        self.get_swap_state().model = False
        self.get_swap_state().ref_model = False
        self.get_swap_state().optimizer = False
        self._loss_fn = get_loss_fn("mlite", self.training_config.loss_func)

    def _build_mlite_config(self):
        return build_mlite_config(self)

    def zero_grad(self) -> None:
        self._require_initialized()
        self.runtime.zero_grad(self.handle)

    def optimizer_step(self) -> Tuple[bool, float, Optional[int]]:
        self._require_initialized()
        update_successful, grad_norm, num_zeros = self.runtime.optimizer_step(self.handle)
        return update_successful, float(grad_norm), num_zeros

    def step_and_get_lr(self) -> float:
        self._require_initialized()
        self.handle._lr_scheduler.step(1)
        return float(self.handle._lr_scheduler.get_last_lr()[0])

    @override
    def finetune_step(
        self,
        batch: List[Dict[str, Any]],
        num_microbatches: int,
        step: int,
    ) -> Dict[str, float]:
        del step
        self._require_initialized()
        self._validate_step_batch(batch, num_microbatches)
        runtime_batches, global_valid_tokens, max_seq_length = (
            self._build_runtime_batches(batch, num_microbatches)
        )
        self.zero_grad()
        with self.runtime.train_mode(self.handle):
            result = self.runtime.forward_backward(
                self.handle,
                iter(runtime_batches),
                loss_fn=self._make_runtime_loss_fn(
                    global_valid_tokens,
                    num_microbatches,
                ),
                num_microbatches=num_microbatches,
                forward_only=False,
            )
        update_successful, grad_norm, _num_zeros = self.optimizer_step()
        if not update_successful:
            raise RuntimeError("mlite optimizer step was skipped or failed")
        lr = self.step_and_get_lr()
        metrics = self._collect_metrics(
            result.metrics,
            global_valid_tokens,
            max_seq_length,
            prefix="finetune",
        )
        metrics["finetune/grad_norm"] = grad_norm
        metrics["finetune/lr"] = lr
        return metrics

    def eval_step(
        self,
        batch: List[Dict[str, Any]],
        num_microbatches: int,
    ) -> Dict[str, float]:
        self._require_initialized()
        self._validate_step_batch(batch, num_microbatches)
        runtime_batches, global_valid_tokens, max_seq_length = (
            self._build_runtime_batches(batch, num_microbatches)
        )
        with self.runtime.eval_mode(self.handle):
            result = self.runtime.forward_backward(
                self.handle,
                iter(runtime_batches),
                loss_fn=self._make_runtime_loss_fn(
                    global_valid_tokens,
                    num_microbatches,
                ),
                num_microbatches=num_microbatches,
                forward_only=True,
            )
        return self._collect_metrics(
            result.metrics,
            global_valid_tokens,
            max_seq_length,
            prefix="eval",
        )

    def _validate_step_batch(
        self,
        batch: List[Dict[str, Any]],
        num_microbatches: int,
    ) -> None:
        expected = num_microbatches * self.training_config.train_mbs
        if len(batch) != expected:
            raise ValueError(f"mlite expected {expected} local samples, got {len(batch)}")

    def _build_runtime_batches(self, batch, num_microbatches):
        return self.prepare_data.sft_to_mlite_packed(
            batch,
            num_microbatches=num_microbatches,
            seq_length=self.training_config.seq_length,
            device=torch.device("cuda", torch.cuda.current_device()),
            dp_size=self.handle.dp_size,
            dp_group=self.handle.dp_group,
        )

    def _make_runtime_loss_fn(
        self,
        global_valid_tokens: torch.Tensor,
        num_microbatches: int,
    ):
        def _runtime_loss_fn(raw_output, runtime_batch, loss_context):
            log_probs = raw_output.get("log_probs")
            if log_probs is None:
                raise ValueError("mlite forward output must contain log_probs")
            protocol = self.handle._extras.get("protocol")
            unpack = getattr(protocol, "unpack_forward_output", None)
            if unpack is None:
                raise RuntimeError("mlite model protocol must provide unpack_forward_output")
            unpacked_log_probs = unpack(
                self._unpack_model(),
                runtime_batch,
                log_probs,
            )
            source_batch = loss_context.source_batch
            loss_result = self._loss_fn(
                self.config,
                MliteFinetuneLossInput(
                    log_probs=unpacked_log_probs,
                    aligned_loss_mask=source_batch["aligned_loss_mask"].values(),
                    global_valid_tokens=global_valid_tokens,
                    dp_size=self.handle.dp_size,
                ),
            )
            metric_loss_sum = loss_result.local_loss_sum
            metric_token_count = loss_result.local_valid_tokens
            if (
                self.policy_config.dist_config.dynamic_context_parallel and
                not runtime_batch.extras["_mlite_dcp_group_leader"]
            ):
                metric_loss_sum = torch.zeros_like(metric_loss_sum)
                metric_token_count = torch.zeros_like(metric_token_count)
            return loss_result.loss * num_microbatches, {
                "_mlite_loss_sum": metric_loss_sum,
                "_mlite_token_count": metric_token_count,
            }

        return _runtime_loss_fn

    def _unpack_model(self):
        model = self.handle._model
        if isinstance(model, (list, tuple)):
            if len(model) == 1:
                return model[0]
            return torch.nn.ModuleList(model)
        return model

    @override
    def setup_model_and_get_optimizer(self) -> int:
        self._mlite_config = self._build_mlite_config()
        self.runtime = create_runtime(
            RuntimeConfig(
                backend="mlite",
                hf_path=self.policy_config.hf_model_path,
                backend_cfg=self._mlite_config,
            )
        )
        self.handle = self.runtime.build_model()
        if self.policy_config.model_arch == "welmv4_moe":
            self._validate_welm_parallel_layout()
        self.swap_impl.bind(self.runtime, self.handle)
        self.model = self._extract_model_chunks()
        self.optimizer = self.handle._optimizer
        if self.optimizer is None:
            raise RuntimeError("mlite FSDP2 runtime did not build an optimizer")
        if self.handle._lr_scheduler is None:
            self.handle._lr_scheduler = build_lr_scheduler(
                self.optimizer,
                self._mlite_config.optimizer,
            )
        if self.handle._lr_scheduler is None:
            raise ValueError(
                "mlite LR scheduler requires training.total_training_step to be positive"
            )
        self.optimizer_scheduler = self.handle._lr_scheduler
        self.get_swap_state().model = True
        self.get_swap_state().optimizer = True

        load_path = self.checkpoint_config.load_ckpt_path
        if not load_path:
            return 0
        # No marker → keep HF weights from build_model (same soft-skip as mcore).
        if not os.path.isfile(os.path.join(load_path, "latest_checkpointed_iteration.txt")):
            return 0
        return self.load_checkpoint(load_path)

    def _extract_model_chunks(self) -> List[torch.nn.Module]:
        model_chunks = self.handle._extras.get("model_chunks")
        if model_chunks is None:
            model = self.handle._model
            model_chunks = list(model) if isinstance(model, (list, tuple)) else [model]
        if not model_chunks:
            raise RuntimeError("mlite runtime returned an empty model chunk list")
        return list(model_chunks)

    def _validate_welm_parallel_layout(self) -> None:
        parallel_state = self.handle._parallel_state
        gcore_dp_size = mpu.get_data_parallel_world_size()
        if gcore_dp_size != parallel_state.dp_size:
            raise RuntimeError(
                "gcore and mlite dense-DP sizes differ for WELM: "
                f"{gcore_dp_size} != {parallel_state.dp_size}"
            )
        if not torch.distributed.is_initialized():
            return
        gcore_dp_ranks = torch.distributed.get_process_group_ranks(mpu.get_data_parallel_group())
        mlite_dp_ranks = torch.distributed.get_process_group_ranks(parallel_state.dp_group)
        if gcore_dp_ranks != mlite_dp_ranks:
            raise RuntimeError(
                "gcore and mlite dense-DP groups differ for WELM: "
                f"{gcore_dp_ranks} != {mlite_dp_ranks}"
            )

    def _require_initialized(self) -> None:
        if self.runtime is None or self.handle is None:
            raise RuntimeError(
                "MliteEngine is not initialized; call setup_model_and_get_optimizer first"
            )

    @override
    def set_model_eval(self) -> None:
        self._require_initialized()
        for chunk in self.model:
            chunk.eval()

    @override
    def set_model_train(self) -> None:
        self._require_initialized()
        for chunk in self.model:
            chunk.train()

    @override
    def offload_model(self) -> None:
        self._require_initialized()
        if not self.get_swap_state().model:
            return
        self.swap_impl.offload_model(self.model)
        self.get_swap_state().model = False
        self.get_swap_state().optimizer = False

    @override
    def onload_model(self) -> None:
        self._require_initialized()
        if self.get_swap_state().model:
            return
        self.swap_impl.onload_model(self.model)
        self.get_swap_state().model = True
        self.get_swap_state().optimizer = True

    def offload_optimizer(self) -> None:
        self._require_initialized()
        super().offload_optimizer()

    def onload_optimizer(self) -> None:
        self._require_initialized()
        super().onload_optimizer()

    def release_grad(self) -> None:
        self._require_initialized()
        super().release_grad()

    @override
    def pretrain_step(
        self,
        batch: List[Dict[str, Any]],
        num_microbatches: int,
        step: int,
    ):
        del batch, num_microbatches, step
        raise NotImplementedError("mlite does not implement pretrain_step")

    @override
    def compute_log_probs(self, rollout_batches, compute_pre_logps=True):
        del rollout_batches, compute_pre_logps
        raise NotImplementedError("mlite finetune does not implement policy log-prob export")

    @override
    def rl_train_actor(self, dataloader_iter):
        del dataloader_iter
        raise NotImplementedError("mlite finetune does not implement RL training")

    def export_weights(self) -> Iterator[Tuple[str, torch.Tensor]]:
        raise NotImplementedError("mlite finetune does not implement rollout weight resync")

    def release_export_scratch(self) -> None:
        raise NotImplementedError("mlite finetune does not implement rollout weight resync")
