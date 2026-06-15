from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from megatron.core import mpu
from megatron.core.utils import divide

try:
    from gpatch_v4.models.deepseek_v4 import DeepseekV4ForCausalLM, apply_hp
except ImportError:
    DeepseekV4ForCausalLM = None
    apply_hp = None
from gpatch_v4.models.hp_module import HpModule
from gpatch_v4.training_backend.fsdp2_backend.checkpoint import (
    save_checkpoint,
    save_hf_checkpoint,
)
from gpatch_v4.training_backend.fsdp2_backend.mtp_loss import calculate_mtp_loss
from gpatch_v4.utils import (
    clear_memory,
    get_batches_max_seqlen,
    get_k_split_list,
    get_max_seqlen_within_dp,
    get_max_seqlen_within_ep,
    log,
)


# reference from slime
def selective_log_softmax_raw(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Fused ``log_softmax → gather``.

    Avoids the memory overhead of allocating a full logprobs tensor.

    Parameters:
        logits: ``[..., V]``.
        input_ids: ``[...]`` token indices.

    Returns:
        ``[...]`` log-probs at ``input_ids``.
    """
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)


selective_log_softmax_compiled = torch.compile(dynamic=True)(selective_log_softmax_raw)


class Fsdp2EngineMixin:
    def _get_vocab_size(self):
        cfg = self.model.config
        if hasattr(cfg, 'vocab_size') and cfg.vocab_size is not None:
            return cfg.vocab_size
        return cfg.get_text_config().vocab_size

    def get_model_cls(self):
        model_arch = getattr(self.policy_config, 'model_arch', None)
        if model_arch == 'gemma4':
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText
        if model_arch == 'deepseek_v4':
            return DeepseekV4ForCausalLM
        return AutoModelForCausalLM

    def _setup_device_mesh(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        self.cp_size = self.config.policy.dist_config.context_parallel_size
        self.dp_size = world_size // self.cp_size
        self.ep_size = self.config.policy.dist_config.expert_model_parallel_size

        # Create 2D device mesh: (dp_size, cp_size)
        # Ranks laid out in row-major: mesh[dp_idx, cp_idx] = dp_idx * cp_size + cp_idx
        # - CP groups: consecutive ranks along dim 1, e.g., [0,1], [2,3], [4,5], [6,7]
        # - DP groups: striped ranks along dim 0, e.g., [0,2,4,6], [1,3,5,7]
        # When cp_size=1, this degenerates to pure DP
        self.mesh = init_device_mesh(
            "cuda", mesh_shape=(self.dp_size, self.cp_size), mesh_dim_names=("dp", "cp")
        )
        self.dp_group = self.mesh.get_group("dp")  # For FSDP gradient sync, metric reduction
        self.cp_group = self.mesh.get_group("cp")  # For Ring Flash Attention, logit gathering
        self.dp_mesh = self.mesh["dp"]

        self.dp_rank = rank // self.cp_size
        self.cp_rank = rank % self.cp_size
        assert self.dp_rank == mpu.get_data_parallel_rank(
        ), f"dp_rank mismatch: {self.dp_rank} vs {mpu.get_data_parallel_rank()}"
        assert self.cp_rank == mpu.get_context_parallel_rank(
        ), f"cp_rank mismatch: {self.cp_rank} vs {mpu.get_context_parallel_rank()}"

        log(
            f"[Rank {rank}] Device mesh (2D): world_size={world_size}, "
            f"cp_size={self.cp_size}, dp_size={self.dp_size}"
        )
        log(
            f"[Rank {rank}] Mesh shape: {self.mesh.shape}, "
            f"dp_rank={self.dp_rank}, cp_rank={self.cp_rank}"
        )

        if self.cp_size > 1:
            # Setup Ring Flash Attention with CP group from mesh (only when cp_size > 1)
            # substitute_hf_flash_attn(self.cp_group, heads_k_stride=1)
            log(f"[Rank {rank}] CP initialized via device mesh")
        else:
            log(f"[Rank {rank}] Pure DP mode (cp_size=1)")

        # ---- HpModule (DSV4 / Qwen3.5-MoE): two independent meshes ----
        # Both span all `world_size` ranks (CP is orthogonal to weight sharding
        # — it doesn't shrink ep_fsdp_size). `apply_hp(model, ep_2d_mesh,
        # cp_mesh=cp_mesh_for_hp)` consumes them directly; the existing
        # `dp_mesh` / `cp_group` above are unused for the HpModule wrap.
        # When `ep_size == 1` / `cp_size == 1` the corresponding axis is just
        # trivial (size-1) and HpModule still works.
        assert world_size % self.ep_size == 0, \
            f"world_size={world_size} not divisible by ep_size={self.ep_size}"
        self.ep_2d_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(world_size // self.ep_size, self.ep_size),
            mesh_dim_names=("ep_fsdp", "ep"),
        )
        cp_full = init_device_mesh(
            "cuda",
            mesh_shape=(world_size // self.cp_size, self.cp_size),
            mesh_dim_names=("dp_for_cp", "cp_hp"),
        )
        self.cp_mesh_for_hp = cp_full["cp_hp"]
        log(
            f"[Rank {rank}] HpModule meshes built: ep_size={self.ep_size}, "
            f"ep_fsdp_size={world_size // self.ep_size}, "
            f"cp_size_hp={self.cp_size}"
        )

        # ---- DSV4 CP × dynamic-pad invariant (fail-fast) ----
        # DSV4's v1 CP (gpatch_v4/models/deepseek_v4/modeling_deepseek_v4.py
        # :1053) requires ``s_local % compress_rate_hca == 0`` where
        # ``compress_rate_hca = max(config.compress_ratios) = 128`` for
        # DSV4-Flash/Pro. Combined with mixin.py:432 dynamic padding
        # (``s_full = pad_to_mulitiple_of × k`` for some ``k ≥ 1``), this
        # forces ``(pad_to_mulitiple_of // cp_size) % 128 == 0``. The
        # ``training.seq_length`` cap at mixin.py:434 must satisfy the same
        # divisibility — otherwise truncation produces an illegal s_local.
        #
        # TODO: when extending HpModule to other architectures, read
        # compress_rate from model config instead of hard-coding 128.
        # The model config isn't loaded yet at this point (model is built
        # later in get_fsdp2_model), so for now we hard-code the DSV4
        # constant and gate by ``model_arch == 'deepseek_v4'``.
        model_arch = getattr(self.policy_config, "model_arch", None)
        if model_arch == "deepseek_v4":
            _COMPRESS_RATE_HCA = 128  # DSV4 HCA m'=128, see config.compress_ratios
            pad_mul = self.training_config.pad_to_mulitiple_of
            seq_len = self.training_config.seq_length
            assert pad_mul % self.cp_size == 0, (
                f"DSV4 CP requires pad_to_mulitiple_of ({pad_mul}) divisible by "
                f"cp_size ({self.cp_size}); otherwise s_full = pad_mul * k may "
                f"not be cp_size-divisible."
            )
            assert (pad_mul // self.cp_size) % _COMPRESS_RATE_HCA == 0, (
                f"DSV4 CP requires (pad_to_mulitiple_of // cp_size) "
                f"({pad_mul} // {self.cp_size} = {pad_mul // self.cp_size}) "
                f"divisible by compress_rate_hca ({_COMPRESS_RATE_HCA}); "
                f"otherwise s_local = (pad_mul * k) / cp_size may not be "
                f"compress_rate-divisible. Bump pad_to_mulitiple_of to a "
                f"multiple of cp_size * {_COMPRESS_RATE_HCA} = "
                f"{self.cp_size * _COMPRESS_RATE_HCA}."
            )
            assert seq_len % self.cp_size == 0, (
                f"DSV4 CP requires training.seq_length ({seq_len}) divisible "
                f"by cp_size ({self.cp_size})."
            )
            assert (seq_len // self.cp_size) % _COMPRESS_RATE_HCA == 0, (
                f"DSV4 CP requires (training.seq_length // cp_size) "
                f"({seq_len} // {self.cp_size} = {seq_len // self.cp_size}) "
                f"divisible by compress_rate_hca ({_COMPRESS_RATE_HCA}); "
                f"otherwise the seq_length cap at mixin.py:434 produces an "
                f"illegal s_local. Bump training.seq_length to a multiple of "
                f"cp_size * {_COMPRESS_RATE_HCA} = "
                f"{self.cp_size * _COMPRESS_RATE_HCA}."
            )

    def _get_init_weight_context_manager(self):
        """Ref: verl/utils/fsdp_utils.py::get_init_weight_context_manager.

        NOTE: tie_word_embedding causes meta_tensor init to hang.
        """
        from accelerate import init_empty_weights

        self._use_meta_tensor = not self.hf_config.tie_word_embeddings

        def cpu_init_weights():
            return torch.device("cpu")

        if self._use_meta_tensor:
            return init_empty_weights if dist.get_rank() != 0 else cpu_init_weights
        else:
            return cpu_init_weights

    def _apply_fsdp2_on_model(self, model, device_mesh):
        layer_cls_to_wrap = model._no_split_modules
        assert len(layer_cls_to_wrap) > 0 and None not in layer_cls_to_wrap

        modules = [
            module for name, module in model.named_modules()
            if module.__class__.__name__ in layer_cls_to_wrap or
            (isinstance(module, torch.nn.Embedding) and not model.config.tie_word_embeddings)
        ]
        param_dtype = torch.bfloat16  # Default to bf16 as before
        reduce_dtype = torch.float32
        fsdp_kwargs = {
            "mp_policy": MixedPrecisionPolicy(
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
            ),
            "mesh": device_mesh,
        }

        for module in modules:
            fully_shard(module, **fsdp_kwargs)
        fully_shard(model, **fsdp_kwargs)

        return model

    def _fsdp2_load_full_state_dict(self, model, full_state):
        if getattr(self, '_use_meta_tensor', True):
            if dist.get_rank() == 0:
                model = model.to(device=torch.cuda.current_device(), non_blocking=True)
            else:
                model = model.to_empty(device=torch.cuda.current_device())
        else:
            model = model.to(device=torch.cuda.current_device(), non_blocking=True)

        is_cpu_offload = False  # 看什么时候想要 offload cpu
        options = StateDictOptions(
            full_state_dict=True, cpu_offload=is_cpu_offload, broadcast_from_rank0=True
        )

        set_model_state_dict(model, full_state, options=options)

        if getattr(self, '_use_meta_tensor', True):
            for _name, buf in model.named_buffers():
                dist.broadcast(buf, src=0)

        if is_cpu_offload:
            model.to("cpu", non_blocking=True)
            for buf in model.buffers():
                buf.data = buf.data.to(torch.cuda.current_device())

        return model

    def get_fsdp2_model(self, init_context, hf_model_path, model_only_inference: bool = False):
        if getattr(self.training_config, "enable_mtp", False):
            assert self.policy_config.model_arch == "deepseek_v4", (
                "FSDP2 MTP finetune is currently only supported for model_arch=deepseek_v4"
            )
        model_cls = self.get_model_cls()
        if not issubclass(model_cls, HpModule):
            # ---- existing stock path unchanged ----
            with init_context():
                model = model_cls.from_pretrained(
                    hf_model_path,
                    trust_remote_code=True,
                    attn_implementation=self.policy_config.attn_implementation,
                )
            if not model_only_inference:
                model.train()
            else:
                model.eval()
            full_state = model.state_dict()

            model = self._apply_fsdp2_on_model(model, device_mesh=self.dp_mesh)
            model = self._fsdp2_load_full_state_dict(model, full_state)

        else:
            # Hybrid-parallel path (DSV4 / Qwen3.5-MoE):
            #   meta-construct → apply_hp (FSDP2 + EP wrap) → load_checkpoint_hp (DSV4)
            #     / load_state_dict_hp (Qwen3.5-MoE, TODO: rename to load_checkpoint_hp)
            #     (per-rank streaming FP8/FP4 dequant for DSV4-Flash)
            # Skip the stock `from_pretrained → push full_state_dict` flow —
            # that would dequantize the entire 480 GB DSV4-Flash on rank 0.
            # `apply_hp` defaults to mp_policy(bf16-fwd, fp32-reduce) and
            # asserts the master is fp32, so meta-construct under fp32
            # default dtype.
            cfg = model_cls.config_class.from_pretrained(hf_model_path, trust_remote_code=True)

            # HP Module 其实不用这个字段，写一个 'eager' fallback 下
            cfg._attn_implementation = 'eager'

            enable_mtp = bool(self.training_config.enable_mtp)
            mtp_num_layers = int(cfg.num_nextn_predict_layers)
            if enable_mtp:
                assert mtp_num_layers > 0, (
                    "training.enable_mtp=True but hf config has no MTP layers "
                    f"(num_nextn_predict_layers={mtp_num_layers})"
                )
                cfg.mtp_loss_scaling_factor = float(
                    getattr(self.training_config, "mtp_loss_scaling_factor", 0.1)
                )
            else:
                cfg.num_nextn_predict_layers = 0

            # DEBUG: optionally truncate to N decoder layers (matches the
            # `_truncate_config` helper in tests/test_gfused/test_deepseek_v4_ep_cp.py).
            # Used for OOM smoke runs; mismatched checkpoint layers are simply
            # ignored by the streaming load path.
            n_layers_dbg = self.config.debug.debug_truncate_num_hidden_layers
            if n_layers_dbg is not None:
                log(
                    f"DEBUG: truncating model config from "
                    f"num_hidden_layers={cfg.num_hidden_layers} -> {n_layers_dbg}",
                    rank=0,
                )
                cfg.num_hidden_layers = n_layers_dbg
                if hasattr(cfg, 'layer_types') and cfg.layer_types is not None:
                    cfg.layer_types = cfg.layer_types[:n_layers_dbg]
                if hasattr(cfg, 'mlp_layer_types') and cfg.mlp_layer_types is not None:
                    cfg.mlp_layer_types = cfg.mlp_layer_types[:n_layers_dbg]
            prev_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float32)
            try:
                with torch.device("meta"):
                    model = model_cls(cfg)
            finally:
                torch.set_default_dtype(prev_dtype)
            model = apply_hp(
                model,
                self.ep_2d_mesh,
                cp_mesh=self.cp_mesh_for_hp,
                attn_backend=self.policy_config.attn_implementation,
                indexer_backend=self.policy_config.indexer_backend,
                ep_backend=self.policy_config.ep_backend,
                deepep_num_sms=self.policy_config.deepep_num_sms,
            )
            model.load_checkpoint_hp(hf_model_path)
            if not model_only_inference:
                model.train()
            else:
                model.eval()

        return model


class CheckpointMixin:
    def save_checkpoint(self, global_step: int):
        if isinstance(self.model, HpModule):
            # HpModule path (DeepSeek-V4 / Qwen3.5-MoE EP+CP+FSDP2): write a
            # self-contained ``from_pretrained``-ready bf16 HF directory under
            # ``<save_ckpt_path>/iter_{step:07d}/hf_hp/``. ``hf_hp`` keeps the
            # subdir distinct from the stock ``model/`` + ``optimizer/`` layout
            # so a follow-up reload won't mis-identify it. ``save_checkpoint_hp``
            # is collective (rank-0 writes, other ranks gather, final barrier),
            # so call it on every rank.
            ck = self.config.checkpoint
            save_path = str(Path(ck.save_ckpt_path).expanduser() / "hf" / f"{global_step}")
            enable_mtp = bool(self.config.training.enable_mtp)
            self.model.save_checkpoint_hp(
                save_path,
                orig_ckpt_dir=self.config.policy.hf_model_path,
                preserve_mtp=not enable_mtp,
            )
            log(f"save_checkpoint_hp wrote {save_path}", rank=0)
            return
        save_checkpoint(
            self.config, self.model, self.optimizer, self.lr_scheduler, global_step=global_step
        )
        if self.config.checkpoint.convert_mcore_to_hf_online:
            save_hf_checkpoint(self.config, self.model, self.tokenizer, global_step=global_step)


class ForwardStepMixin:
    @torch.no_grad()
    def compute_logprobs(
        self, model, batches_list: List[Dict[str, Any]], batch_log_str: str
    ) -> torch.Tensor:
        total_samples = len(batches_list)
        seq_length = get_batches_max_seqlen(batches_list, self.training_config.pad_to_mulitiple_of)
        seq_length = get_max_seqlen_within_ep(seq_length)
        num_microbatches = divide(total_samples, self.forward_only_mbs)
        batch_iter = get_k_split_list(batches_list, num_microbatches)

        logprobs_list = []

        for batches in tqdm(batch_iter, desc=batch_log_str, disable=dist.get_rank() != 0):
            model_fwd_args = self.prepare_data.model_forward_only(
                batches,
                seq_length,
                self.tokenizer.pad_token_id,
                pad_with_random_token=self.config.training.moe_pad_with_random_tokens,
                vocab_size=self._get_vocab_size(),
            )
            target = model_fwd_args.pop("target")
            logits = model(**model_fwd_args).logits.float()

            logprobs, _ = self.get_logprob_and_entropy(
                logits=logits,
                target_tokens=target,
                allow_compile=False,
                temperature=None,
                return_entropy=False,
            )
            logprobs_list.append(logprobs)

        logprobs = torch.cat(logprobs_list) if len(logprobs_list) > 0 else None
        assert logprobs.shape[0] == total_samples
        logprobs = [logprob.squeeze(0) for logprob in logprobs.cpu().chunk(total_samples)]
        clear_memory()
        return logprobs

    def gather_log_probs_packed(
        self,
        shifted_logits: torch.Tensor,
        input_ids: torch.Tensor,
        allow_compile: bool,
        cu_seqlens: torch.Tensor | float | None = None,
        temperature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gather next-token log probs for packed sequences.

        Parameters:
            logits: ``[B, S, V]``.
            input_ids: ``[B, S]``.
            cu_seqlens: unused (kept for API compat).

        Returns:
            ``[B, S-1]`` target log-probs.
        """
        # Handle batch dimension - logits should be [batch_size, seq_len, vocab_size]
        assert shifted_logits.dim() == 3
        assert input_ids.dim() == 2
        assert cu_seqlens is None, "cu_seqlens is not supported"

        if temperature is not None:
            shifted_logits = shifted_logits.div(temperature)

        targets = input_ids[:, 1:].to(device=shifted_logits.device)
        assert shifted_logits.shape[:2
                                   ] == targets.shape, f"{shifted_logits.shape=} {targets.shape=}"

        # Gather log probs for targets
        selective_log_softmax = selective_log_softmax_compiled if allow_compile else selective_log_softmax_raw
        return selective_log_softmax(shifted_logits, targets)

    def get_logprob_and_entropy(
        self,
        logits: torch.Tensor,
        target_tokens: torch.Tensor,
        allow_compile: bool,
        temperature: float | None = None,
        return_entropy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute log probabilities and entropy.

        Parameters:
            logits: ``[B, S, V]``.
            target_tokens: ``[B, S]``.
            allow_compile:
            temperature: optional.

        Returns:
            log_probs: ``[B, S-1]``.
            entropy: ``[B, S-1]``.
        """
        shifted_logits = logits[:, :-1, :]
        log_probs = self.gather_log_probs_packed(
            shifted_logits, target_tokens, allow_compile=allow_compile, temperature=temperature
        )
        if return_entropy:
            log_probs_full = torch.log_softmax(shifted_logits, dim=-1)
            probs = torch.softmax(shifted_logits, dim=-1)
            entropy = -(probs * log_probs_full).sum(dim=-1)
        else:
            entropy = None
        return log_probs, entropy

    def _calc_dynamic_mbs(self, len_batch, max_seq_length):
        dynamic_mbs = self.config.training.seq_length // max_seq_length
        while dynamic_mbs >= 1:
            if len_batch % dynamic_mbs == 0:
                break
            dynamic_mbs -= 1
        assert dynamic_mbs >= 1
        return dynamic_mbs

    def _finetune_step(
        self, batch: List[Dict[str, Any]], num_microbatches: int, forward_only: bool = False
    ):
        training_config = self.config.training
        dp_size = mpu.get_data_parallel_world_size()
        assert training_config.train_gbs == len(
            batch
        ) * dp_size, f"{training_config.train_gbs=} {len(batch)=} {dp_size=}"

        if self.config.debug.experimental_pad_to_max_length:
            max_seq_length = self.config.training.seq_length
        else:
            max_seq_length = get_batches_max_seqlen(batch, self.training_config.pad_to_mulitiple_of)
            max_seq_length = get_max_seqlen_within_dp(max_seq_length)
            max_seq_length = min(max_seq_length, training_config.seq_length)
        if training_config.loss_func in ["square_averaging_cross_entropy"]:
            assert False
            update_square_averaging_token_len(batch, max_seq_length)

        dynamic_mbs = 1
        if training_config.use_dynamic_mbs:
            dynamic_mbs = self._calc_dynamic_mbs(len(batch), max_seq_length)
            num_microbatches = len(batch) // dynamic_mbs

        data_iter = get_k_split_list(batch, num_microbatches)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        report_loss = 0.
        report_main_loss = 0.
        report_mtp_loss = 0.
        report_mtp_depth_loss = None
        for batches in tqdm(data_iter, disable=True):
            batch, fwd_kwargs = self.prepare_data.sft_train(
                batches,
                max_seq_length,
                self.tokenizer.pad_token_id,
                comput_attn_mask=self.training_config.comput_attn_mask,
                pad_with_random_token=self.config.training.moe_pad_with_random_tokens,
                input_teacher_logits=getattr(self.config.training, "enable_teacher_kl_loss", False),
                vocab_size=self._get_vocab_size(),
            )

            loss_mask = batch["loss_mask"]
            labels = batch["labels"]
            # Cast to fp32 for numerically stable cross entropy (log-sum-exp in
            # bf16/fp16 is lossy); consistent with logprob path above.
            outputs = self.model(**fwd_kwargs)
            logits = outputs.logits.float()
            labels_2d = labels
            loss_mask_2d = loss_mask

            # tokens 和 logits 都在 sft_train 时候 shift 过了
            logits = logits.view(-1, self._get_vocab_size())
            labels = labels_2d.view(-1)
            loss_mask = loss_mask_2d.view(-1)

            # Total valid tokens across DP*CP world.
            global_n = (labels != -100).sum()
            dist.all_reduce(global_n)

            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            loss = loss * loss_mask.to(loss.device)

            # TODO: 将 loss func 独立出去
            # Per-token mean (= megatron's sum(loss)/sum(loss_mask)). `* dp_size`
            # cancels the 1/dp_size from the final WORLD-AVG on `report_loss`
            # (CP is folded by the CP-SUM on `tmp`, so only DP needs compensation).
            main_loss = torch.sum(loss) / (global_n + 1e-8)
            main_loss = main_loss * self.dp_size

            mtp_loss = None
            mtp_depth_losses = None
            if training_config.enable_mtp:
                mtp_per_depth_h = getattr(outputs, "mtp_per_depth_h", None)
                assert mtp_per_depth_h is not None, (
                    "enable_mtp=True but model forward returned no mtp_per_depth_h"
                )
                model_cp_group = self.model._cp_group if self.cp_size > 1 else None
                mtp_scale = float(
                    getattr(
                        outputs,
                        "mtp_loss_scaling_factor",
                        getattr(training_config, "mtp_loss_scaling_factor", 0.1),
                    )
                )
                _, _, mtp_depth_nums, mtp_depth_dens = calculate_mtp_loss(
                    mtp_per_depth_h=mtp_per_depth_h,
                    labels=labels_2d.to(logits.device),
                    lm_head=self.model.lm_head,
                    loss_fct=loss_fct,
                    loss_mask=loss_mask_2d.to(logits.device),
                    cp_group=model_cp_group,
                    scaling_factor=mtp_scale,
                )
                mtp_depth_losses = []
                mtp_depth_loss_metrics = []
                for n_local, d_local in zip(mtp_depth_nums, mtp_depth_dens):
                    d_global = d_local.detach().clone()
                    # Align MTP normalization with main_loss:
                    # use WORLD valid-token denominator, keep local numerator
                    # for backward scaling, then compensate FSDP's dp averaging
                    # via `* dp_size` (same as main_loss).
                    dist.all_reduce(d_global)
                    mtp_depth_losses.append((n_local / d_global.clamp_min(1.0)) * self.dp_size)
                    n_global_metric = n_local.detach().clone()
                    # Reporting metric: global numerator/global denominator.
                    dist.all_reduce(n_global_metric)
                    mtp_depth_loss_metrics.append(n_global_metric / d_global.clamp_min(1.0))
                mtp_loss = torch.stack(mtp_depth_losses
                                      ).sum() * (mtp_scale / max(len(mtp_depth_losses), 1))
                loss = main_loss + mtp_loss
            else:
                loss = main_loss

            # TODO: still biased across micro batches with different
            # valid-token counts (each micro normalizes by its own global_n).
            loss = loss / num_microbatches
            if not forward_only:
                loss.backward()

            tmp = loss.detach().clone()
            tmp_main = main_loss.detach().clone() / num_microbatches
            if self.cp_size > 1:
                dist.all_reduce(tmp, group=self.model._cp_group)
                dist.all_reduce(tmp_main, group=self.model._cp_group)
            report_loss += tmp.item()
            report_main_loss += tmp_main.item()
            if mtp_loss is not None:
                tmp_mtp = mtp_loss.detach().clone() / num_microbatches
                if self.cp_size > 1:
                    dist.all_reduce(tmp_mtp, group=self.model._cp_group)
                report_mtp_loss += tmp_mtp.item()
                if report_mtp_depth_loss is None:
                    report_mtp_depth_loss = [0.0 for _ in range(len(mtp_depth_losses))]
                for i, dloss in enumerate(mtp_depth_loss_metrics):
                    dtmp = dloss.detach().clone() / num_microbatches
                    report_mtp_depth_loss[i] += dtmp.item()

        report_loss = torch.tensor(report_loss).to(torch.cuda.current_device())
        torch.distributed.all_reduce(report_loss, op=torch.distributed.ReduceOp.AVG)
        report_main_loss_t = torch.tensor(report_main_loss).to(torch.cuda.current_device())
        torch.distributed.all_reduce(report_main_loss_t, op=torch.distributed.ReduceOp.AVG)
        report_mtp_loss_t = torch.tensor(report_mtp_loss).to(torch.cuda.current_device())
        torch.distributed.all_reduce(report_mtp_loss_t, op=torch.distributed.ReduceOp.AVG)

        metric_prefix = "finetune"
        if forward_only:
            metric_prefix = "eval"
        metrics = {
            f"{metric_prefix}/lm_loss": report_main_loss_t.item(),
            f"{metric_prefix}/total_loss": report_loss.item(),
            f"{metric_prefix}/seq_length": max_seq_length,
        }
        if training_config.enable_mtp:
            metrics[f"{metric_prefix}/mtp_loss"] = report_mtp_loss_t.item()
            if report_mtp_depth_loss is not None:
                for i, value in enumerate(report_mtp_depth_loss):
                    depth_t = torch.tensor(value).to(torch.cuda.current_device())
                    torch.distributed.all_reduce(depth_t, op=torch.distributed.ReduceOp.AVG)
                    metrics[f"{metric_prefix}/mtp_depth_{i}_loss"] = depth_t.item()
        if training_config.use_dynamic_mbs:
            metrics[f"{metric_prefix}/dynamic_mbs"] = dynamic_mbs
        return metrics

    def clip_grad_norm_(self):
        if not isinstance(self.model, HpModule):
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.config.optimizer.max_grad_norm
            )
            grad_norm = grad_norm.full_tensor()
            return grad_norm
        else:
            # `apply_hp` binds an EP-aware clip_grad_norm_ that aggregates
            # expert grad norms across the EP group before clipping. It
            # returns a Python float (`total_norm.item()` in
            # `_clip_grad_norm_multi_mesh`); the caller in
            # `fsdp2_engine_lm.py` expects a Tensor (does
            # `torch.isfinite(grad_norm)`), so wrap in a 0-d cuda tensor
            # to match the stock-path contract.
            gn = self.model.clip_grad_norm_(self.config.optimizer.max_grad_norm)
            return torch.tensor(float(gn), device=torch.cuda.current_device())
