import uuid
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
from typing_extensions import override

from megatron.core import mpu, parallel_state
from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler
from megatron.core.datasets.data_schedule_utils import _get_global_seqlens_and_ids
from megatron.core.extensions.transformer_engine import get_thd_partitioned_indices
from megatron.core.packed_seq_params import PackedSeqParams

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.core.parallel_state import is_mp_and_cp_head
from gpatch_v4.extended_model.base import PrepareDataForward, SamplerGenerateFunc
from gpatch_v4.extended_model.mtp_mixin import OnlineMtpSftMixin
from gpatch_v4.generation_backend.routed_experts_utils import process_routed_experts
from gpatch_v4.utils import (
    BroadcastUtils,
    get_ltor_masks_and_position_ids,
    get_tensor_on_this_cp_rank,
    pad_3d_seq_dim,
    pad_or_truncate_last_dim,
)
from gpatch_v4.utils.dynamic_cp_utils import (
    _round_up,
    sft_dyn_cp_schedule_default,
    sft_dyn_cp_schedule_smart_padding,
)


class SamplerGenerateFuncLLM(SamplerGenerateFunc):
    """LLM-specific sampler generation function."""
    @override
    async def __call__(self, config, infer_engine, idx, tokenizer, batched_data,
                       sampling_repeat_n) -> Dict[str, List[Any]]:
        rank_unique_ids = batched_data["unique_id"]
        prompt_token_ids = batched_data["prompt_token_ids"]
        prompt_lens = batched_data["prompt_lens"]
        gt_label = batched_data["gt_label"]

        sampling_params = infer_engine.get_sampling_params_from_config(
            config.sampler.infer_engine_configs[idx], tokenizer.eos_token_id
        )
        async_gens = []
        for i in range(len(prompt_token_ids)):
            for j in range(sampling_repeat_n):
                tmp_sampling_params = infer_engine.copy_sampling_params_with_seed_offset(
                    sampling_params, i * sampling_repeat_n + j
                )
                gen = infer_engine.async_generate(
                    prompt_token_ids[i],
                    tmp_sampling_params,
                    str(uuid.uuid4().hex),
                    return_routed_experts=config.training.moe_router_replay,
                )
                async_gens.append(gen)

        gen_outputs = await infer_engine.wait_and_get_async_generate_output(async_gens)

        tokens_lst = []
        seq_length_lst = []
        prompt_len_lst = []
        gt_label_lst = []
        routed_experts_list = []
        unique_id_list = []
        rollout_log_probs_lst = []

        num_layers = config.policy.hf_config.num_hidden_layers
        moe_router_topk = getattr(config.policy.hf_config, "num_experts_per_tok", None)

        for gi, gen_out in enumerate(gen_outputs):
            i = gi // sampling_repeat_n
            j = gi % sampling_repeat_n
            assert len(gen_out.outputs) == 1

            one_output = gen_out.outputs[0]
            resp_tokens = list(one_output.token_ids)
            one_prompt_token_ids = prompt_token_ids[i]['prompt_token_ids']
            token = one_prompt_token_ids + resp_tokens
            assert len(token) <= config.training.seq_length
            tokens_lst.append(torch.tensor(token, dtype=torch.long))
            seq_length_lst.append(torch.tensor(len(token), dtype=torch.long))
            prompt_len_lst.append(prompt_lens[i])
            gt_label_lst.append(gt_label[i])
            unique_id_list.append(rank_unique_ids[i])

            rollout_log_prob = one_output.output_logprobs
            assert len(resp_tokens
                      ) == len(rollout_log_prob), f"{len(resp_tokens)=} {len(rollout_log_prob)=}"
            gen_lp = torch.tensor(rollout_log_prob, dtype=torch.float32)
            prompt_len = len(one_prompt_token_ids)
            full_lp = torch.ones(len(token), dtype=torch.float32)
            gen_len = gen_lp.size(0)
            assert len(token) == prompt_len + gen_len
            full_lp[prompt_len - 1:prompt_len + gen_len - 1] = gen_lp
            rollout_log_probs_lst.append(full_lp)

            routed_experts = process_routed_experts(one_output, num_layers, moe_router_topk)
            routed_experts_list.append(routed_experts)

        rollout_batch = dict(
            tokens=tokens_lst,
            sequence_lengths=seq_length_lst,
            prompt_lengths=prompt_len_lst,
            gt_label=gt_label_lst,
            unique_id=unique_id_list,
            rollout_log_probs=rollout_log_probs_lst,
        )
        if config.training.moe_router_replay:
            rollout_batch["routed_experts"] = routed_experts_list
        return rollout_batch


class PrepareDataForwardLLM(OnlineMtpSftMixin, PrepareDataForward):
    @override
    def model_forward_only(
        self,
        batches: List[Dict[str, Any]],
        seqlen: int,
        pad_token_id: int,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        model_fwd_args = {}
        vocab_size = kwargs.get("vocab_size", 0)

        tokens_l = []
        for batch in batches:
            tokens_l.append(
                pad_or_truncate_last_dim(
                    batch["tokens"],
                    seqlen,
                    pad_token_id,
                    pad_with_random_token=pad_with_random_token,
                    vocab_size=vocab_size,
                )
            )

        tokens = torch.stack(tokens_l).view(len(tokens_l), -1).cuda(non_blocking=True)
        model_fwd_args["target"] = tokens.detach().clone()

        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
            tokens, 0, False, False, False, compute_attention_mask=False
        )
        if attention_mask is not None:
            attention_mask = attention_mask.expand(tokens.size(0), -1, -1, -1)

        if dist.get_world_size(mpu.get_context_parallel_group()) > 1:
            tokens = get_tensor_on_this_cp_rank(tokens, 1, key_name="tokens")
            attention_mask = get_tensor_on_this_cp_rank(
                attention_mask, 2, key_name="attention_mask"
            )
            position_ids = get_tensor_on_this_cp_rank(position_ids, 1, key_name="position_ids")

        model_fwd_args["input_ids"] = tokens
        model_fwd_args["position_ids"] = position_ids
        model_fwd_args["attention_mask"] = attention_mask
        return model_fwd_args

    @override
    def grpo_train(
        self,
        batches: List[Dict[str, Any]],
        seqlen: int,
        pad_token_id: int,
        ppo_pack_seq: bool,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        non_blocking = True
        vocab_size = kwargs.get("vocab_size", 0)
        tokens_l = []
        advantages_l = []
        mask_l = []
        logprobs_l = []
        ref_logprobs_l = []
        rollout_logprobs_l = []
        sequence_lengths_l = []
        has_rollout_logprobs = "rollout_log_probs" in batches[0]
        has_ref_logprobs = "ref_logprobs" in batches[0]
        for batch in batches:
            tokens_l.append(
                pad_or_truncate_last_dim(
                    batch['tokens'],
                    seqlen,
                    pad_token_id,
                    pad_with_random_token=pad_with_random_token,
                    vocab_size=vocab_size,
                )
            )
            advantages_l.append(pad_or_truncate_last_dim(batch['advantages'], seqlen - 1, 0))
            mask_l.append(pad_or_truncate_last_dim(batch['mask'], seqlen - 1, 0))
            logprobs_l.append(pad_or_truncate_last_dim(batch['logprobs'], seqlen - 1, 0))
            if has_ref_logprobs:
                ref_logprobs_l.append(
                    pad_or_truncate_last_dim(batch['ref_logprobs'], seqlen - 1, 0)
                )
            if has_rollout_logprobs:
                rollout_logprobs_l.append(
                    pad_or_truncate_last_dim(batch['rollout_log_probs'], seqlen - 1, 0)
                )
            sequence_lengths_l.append(batch['sequence_lengths'])

        tokens = torch.stack(tokens_l).cuda(non_blocking=non_blocking)
        advantages = torch.stack(advantages_l)
        mask = torch.stack(mask_l)
        logprobs = torch.stack(logprobs_l)
        ref_logprobs = torch.stack(ref_logprobs_l) if has_ref_logprobs else None
        rollout_log_probs = torch.stack(rollout_logprobs_l) if has_rollout_logprobs else None
        sequence_lengths = torch.stack(sequence_lengths_l)

        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
            data=tokens,
            eod_token=0,  # unused
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            compute_attention_mask=False,
        )
        target = tokens.detach().clone()

        # mtp requires positon_ids and loss_mask
        mtp_labels, mtp_loss_mask = self._build_online_mtp_labels(
            tokens, mask, do_cp_split=not ppo_pack_seq
        )

        if dist.get_world_size(mpu.get_context_parallel_group()) > 1 and not ppo_pack_seq:
            tokens = get_tensor_on_this_cp_rank(tokens, 1, key_name="tokens")
            attention_mask = get_tensor_on_this_cp_rank(
                attention_mask, 2, key_name="attention_mask"
            )
            position_ids = get_tensor_on_this_cp_rank(position_ids, 1, key_name="position_ids")

        batch = {
            "tokens": tokens,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "advantages": advantages,
            "prev_log_probs": logprobs,
            "mask": mask,
            "ref_log_probs": ref_logprobs,
            'target': target,
            'sequence_lengths': sequence_lengths,
            'mtp_labels': mtp_labels,
            'mtp_loss_mask': mtp_loss_mask,
        }
        if has_rollout_logprobs:
            batch["rollout_log_probs"] = rollout_log_probs

        if "sample_mask" in batches[0]:
            sample_mask_l = []
            for _batch in batches:
                sample_mask_l.append(_batch['sample_mask'])
            batch['sample_mask'] = torch.stack(sample_mask_l).cuda()

        if "global_retention_ratio" in batches[0]:
            batch['global_retention_ratio'] = batches[0]['global_retention_ratio'].cuda()

        required_keys = set()
        if mpu.get_pipeline_model_parallel_world_size() == 1:
            required_keys.update(batch.keys())
        else:
            required_keys.add("attention_mask")
            required_keys.add("sequence_lengths")
            required_keys.add("position_ids")
            if mpu.is_pipeline_first_stage():
                required_keys.update(("tokens", ))
            if mpu.is_pipeline_last_stage():
                required_keys.update(
                    (
                        "tokens", "advantages", "mask", "prev_log_probs", "ref_log_probs",
                        "rollout_log_probs", 'target', 'sample_mask', 'global_retention_ratio'
                    )
                )
                # mtp requires positon_ids and labels
                if self.config.training.online_mtp_sft:
                    required_keys.add("position_ids")
                    required_keys.add("mtp_labels")
                    required_keys.add("mtp_loss_mask")

        batch = {
            key:
                (
                    val.cuda(non_blocking=non_blocking)
                    if key in required_keys and val is not None else None
                )
            for key, val in batch.items()
        }

        fwd_kwargs = dict(
            input_ids=batch.pop("tokens"),
            position_ids=batch.pop("position_ids"),
            attention_mask=batch.pop("attention_mask"),
            labels=batch.get("mtp_labels", None),
            loss_mask=batch.get("mtp_loss_mask", None),
        )
        return batch, fwd_kwargs

    @override
    def ppo_value_train(
        self,
        batches: List[Dict[str, Any]],
        seqlen: int,
        pad_token_id: int,
        ppo_pack_seq: bool,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        non_blocking = True
        vocab_size = kwargs.get("vocab_size", 0)
        tokens_l = []
        values_l = []
        returns_l = []
        mask_l = []
        sequence_lengths_l = []
        for batch in batches:
            tokens_l.append(
                pad_or_truncate_last_dim(
                    batch['tokens'],
                    seqlen,
                    pad_token_id,
                    pad_with_random_token=pad_with_random_token,
                    vocab_size=vocab_size,
                )
            )
            values_l.append(pad_or_truncate_last_dim(
                batch['values'],
                seqlen - 1,
                0.0,
            ))
            returns_l.append(pad_or_truncate_last_dim(
                batch['returns'],
                seqlen - 1,
                0.0,
            ))
            mask_l.append(pad_or_truncate_last_dim(batch['mask'], seqlen - 1, 0))
            sequence_lengths_l.append(batch['sequence_lengths'])

        tokens = torch.stack(tokens_l).cuda(non_blocking=non_blocking)
        mask = torch.stack(mask_l)
        values = torch.stack(values_l).cuda(non_blocking=non_blocking)
        returns = torch.stack(returns_l).cuda(non_blocking=non_blocking)
        sequence_lengths = torch.stack(sequence_lengths_l).cuda(non_blocking=non_blocking)
        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
            data=tokens,
            eod_token=0,  # unused
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            compute_attention_mask=False,
        )

        # 数据先被 pad to multiple of 过了
        if dist.get_world_size(mpu.get_context_parallel_group()) > 1 and not ppo_pack_seq:
            tokens = get_tensor_on_this_cp_rank(tokens, 1, key_name="tokens")
            attention_mask = get_tensor_on_this_cp_rank(
                attention_mask, 2, key_name="attention_mask"
            )
            position_ids = get_tensor_on_this_cp_rank(position_ids, 1, key_name="position_ids")

        batch = {
            "tokens": tokens,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "mask": mask,
            "values": values,
            'sequence_lengths': sequence_lengths,
            'returns': returns,
        }
        required_keys = set()
        if mpu.get_pipeline_model_parallel_world_size() == 1:
            required_keys.update(batch.keys())
        else:
            required_keys.add("attention_mask")
            required_keys.add("sequence_lengths")
            required_keys.add("position_ids")
            if mpu.is_pipeline_first_stage():
                required_keys.update(("tokens", ))
            if mpu.is_pipeline_last_stage():
                required_keys.update(("tokens", "mask", "values", "returns"))

        batch = {
            key:
                (
                    val.cuda(non_blocking=non_blocking)
                    if key in required_keys and val is not None else None
                )
            for key, val in batch.items()
        }

        fwd_kwargs = dict(
            input_ids=batch.pop("tokens"),
            position_ids=batch.pop("position_ids"),
            attention_mask=batch.pop("attention_mask"),
            labels=None,
        )
        return batch, fwd_kwargs

    def _prepare_tokens_and_labels(
        self,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        seq_len: int,
        pad_token_id: int,
        vocab_size: int,
        pad_with_random_token: bool = False,
    ):
        # 先判断 labels 是否有被 shift 过
        assert tokens.shape == labels.shape, f"{tokens.shape=}, {labels.shape=}"
        assert torch.equal(
            tokens == labels, labels >= 0
        ), f"labels should not be shifted:{tokens.tolist()=} {labels.tolist()=}"
        if tokens.shape[-1] <= seq_len:
            actual_len = tokens.shape[-1]
            # 多加一位是为了 shift
            tokens = pad_or_truncate_last_dim(
                tokens,
                seq_len + 1,
                pad_token_id,
                pad_with_random_token=pad_with_random_token,
                vocab_size=vocab_size,
            )
            labels = pad_or_truncate_last_dim(labels, seq_len + 1, -100)
            tokens = tokens[:-1]
            labels = labels[1:]
        else:
            tokens = tokens[:-1]
            labels = labels[1:]
            tokens = tokens[-seq_len:]
            labels = labels[-seq_len:]
            actual_len = tokens.shape[-1]
        return tokens, labels, actual_len

    def _sft_train_cp_chunk_data(
        self,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: None | torch.Tensor,
        attention_mask: None | torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-rank CP slicing of sft_train tensors.

        Default: Megatron-style zigzag slicing along the sequence axis. Sub-
        classes whose downstream pipeline performs CP slicing differently
        (e.g. DSV4 / Qwen3.5-MoE running as ``HpModule`` where the subclass
        contiguously slices ``tokens`` / ``labels`` / ``loss_mask`` /
        ``position_ids`` in one call to
        :func:`gpatch_v4.models.deepseek_v4.cp.cp_chunk_data`) should
        override this. Called from ``sft_train`` only when
        ``cp_world_size > 1``.
        """
        tokens = get_tensor_on_this_cp_rank(tokens, 1, key_name="tokens")
        labels = get_tensor_on_this_cp_rank(labels, 1, key_name="labels")
        loss_mask = get_tensor_on_this_cp_rank(loss_mask, 1, key_name="loss_mask")
        position_ids = get_tensor_on_this_cp_rank(position_ids, 1, key_name="position_ids")
        if attention_mask is not None:
            attention_mask = get_tensor_on_this_cp_rank(
                attention_mask, 2, key_name="attention_mask"
            )
        return tokens, labels, loss_mask, position_ids, attention_mask

    @override
    def sft_train(
        self,
        batches: List[Dict[str, Any]],
        seq_len: int,
        pad_token_id: int,
        comput_attn_mask: bool = True,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        vocab_size = kwargs.get("vocab_size", 0)
        token_list = []
        label_list = []
        square_averaging_weight_list = []
        for i, batch in enumerate(batches):
            token, label, _ = self._prepare_tokens_and_labels(
                batch["tokens"],
                batch["labels"],
                seq_len,
                pad_token_id,
                vocab_size,
                pad_with_random_token,
            )

            token_list.append(token)
            label_list.append(label)
            if "square_averaging_weight" in batch:
                square_averaging_weight_list.append(batch["square_averaging_weight"])
        tokens = torch.stack(token_list).view(len(token_list), -1).cuda(non_blocking=True)
        labels = torch.stack(label_list).view(len(token_list), -1).cuda(non_blocking=True)
        square_averaging_weights = None
        if len(square_averaging_weight_list) > 0:
            assert len(square_averaging_weight_list) == len(token_list)
            square_averaging_weights = torch.stack(square_averaging_weight_list).view(
                len(square_averaging_weight_list), -1
            ).cuda(non_blocking=True)

        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            labels, -100, False, False, True, compute_attention_mask=comput_attn_mask
        )
        full_loss_mask = loss_mask

        if dist.get_world_size(mpu.get_context_parallel_group()) > 1:
            tokens, labels, loss_mask, position_ids, attention_mask = (
                self._sft_train_cp_chunk_data(
                    tokens, labels, loss_mask, position_ids, attention_mask
                )
            )

        batch = {
            "tokens": tokens,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "full_loss_mask": full_loss_mask,
            "square_averaging_weights": square_averaging_weights,
        }

        # check all on gpu
        assert all([x.is_cuda for x in batch.values() if x is not None])
        fwd_kwargs = dict(
            input_ids=batch.pop("tokens"),
            position_ids=batch.pop("position_ids"),
            attention_mask=batch.pop("attention_mask"),
            labels=None,
        )
        # labels/loss_mask are already CP-split above; let the model compute the
        # MTP loss from them when online_mtp_sft is enabled.
        self._maybe_set_mtp_sft_fwd_kwargs(fwd_kwargs, batch["labels"], batch["loss_mask"])
        return batch, fwd_kwargs

    @override
    def sft_train_with_dynamic_cp(
        self,
        batches: List[Dict[str, Any]],
        seq_len: int,
        pad_token_id: int,
        comput_attn_mask: bool = True,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        assert len(batches) == 1, "sft_train_with_dynamic_cp only supports one batch"
        batch = batches[0]
        assert "local_cp_size" in batch

        # 1. get cp_group
        lcp = batch.get("local_cp_size")
        if lcp is not None:
            lcp_val = lcp.item() if isinstance(lcp, torch.Tensor) else int(lcp)
            cp_group = parallel_state.get_dynamic_data_context_parallel_groups(group_size=lcp_val)
        else:
            cp_group = parallel_state.get_context_parallel_group()

        # 2. do cp slicing (THD load balancing)
        cp_size = cp_group.size()
        if cp_size > 1:
            cp_rank = cp_group.rank()
            total_tokens = batch["tokens"].size(0)
            # Pass cu_seqlens_padded as cu_seqlens to work around a TE bug
            # in thd_get_partitioned_indices.
            cu_seqlens_for_partition = batch["cu_seqlens_padded"]
            index = get_thd_partitioned_indices(
                cu_seqlens_for_partition, total_tokens, cp_size, cp_rank
            )
            for key in ["tokens", "labels", "loss_mask", "position_ids"]:
                assert key in batch
                batch[key] = batch[key].index_select(0, index)

        # 3. align tp
        tp_size = parallel_state.get_tensor_model_parallel_group().size()
        assert batch["tokens"].size(0) % tp_size == 0, (
            f"post-CP tokens ({batch['tokens'].size(0)}) not aligned to tp_size={tp_size}"
        )

        # 4. change view
        total_tokens_val = torch.tensor(batch["tokens"].size(0), dtype=torch.int32)
        batch["tokens"] = batch["tokens"].view(1, total_tokens_val).contiguous()
        batch["position_ids"] = batch["position_ids"].view(1, total_tokens_val).contiguous()
        batch["labels"] = batch["labels"].view(1, total_tokens_val).contiguous()
        batch["loss_mask"] = batch["loss_mask"].view(1, total_tokens_val).contiguous()

        cu_seqlens_padded = batch["cu_seqlens_padded"]
        max_seqlen = batch["max_seqlen"].item()
        local_cp_size = batch["local_cp_size"].item()
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens_padded,
            cu_seqlens_kv=cu_seqlens_padded,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            local_cp_size=local_cp_size,
            cp_group=cp_group,
        )

        fwd_kwargs = dict(
            input_ids=batch["tokens"],
            position_ids=batch["position_ids"],
            attention_mask=None,
            labels=None,
            packed_seq_params=packed_seq_params,
        )

        # Store cp_group in batch so the loss function can use it for CP reduction.
        batch["cp_group"] = cp_group

        return batch, fwd_kwargs

    @override
    def opd_train(
        self,
        batches: List[Dict[str, Any]],
        seqlen: int,
        pad_token_id: int,
        ppo_pack_seq: bool,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        non_blocking = True
        vocab_size = kwargs.get("vocab_size", 0)
        tokens_l = []
        advantages_l = []
        mask_l = []
        logprobs_l = []
        ref_logprobs_l = []
        teacher_logprobs_l = []
        rollout_logprobs_l = []
        sequence_lengths_l = []
        stu_topk_logprobs_l = []
        stu_topk_ids_l = []
        has_rollout_logprobs = "rollout_log_probs" in batches[0]
        has_ref_logprobs = "ref_logprobs" in batches[0]
        has_topk = "stu_topk_logprobs" in batches[0] and "stu_topk_ids" in batches[0]

        teacher_names = list(self.config.teachers.keys())
        is_single_teacher = len(teacher_names) == 1
        routing_field = getattr(self.config.ppo, "g_opd_teacher_routing_field", "teacher_type")
        for bi, batch in enumerate(batches):
            tokens_l.append(
                pad_or_truncate_last_dim(
                    batch['tokens'],
                    seqlen,
                    pad_token_id,
                    pad_with_random_token=pad_with_random_token,
                    vocab_size=vocab_size,
                )
            )
            adv = batch['advantages']
            if adv.dim() == 2:
                advantages_l.append(pad_3d_seq_dim(adv, seqlen - 1, 0))
            else:
                advantages_l.append(pad_or_truncate_last_dim(adv, seqlen - 1, 0))
            mask_l.append(pad_or_truncate_last_dim(batch['mask'], seqlen - 1, 0))
            logprobs_l.append(pad_or_truncate_last_dim(batch['logprobs'], seqlen - 1, 0))

            if is_single_teacher:
                teacher_name = teacher_names[0]
            else:
                assert routing_field in batch, (
                    f"multi-teacher opd requires per-sample routing field "
                    f"'{routing_field}' in batch, got keys={list(batch.keys())}"
                )
                teacher_name = batch[routing_field]
                assert teacher_name in teacher_names, (
                    f"sample routed to teacher '{teacher_name}' which is not in "
                    f"configured teachers {teacher_names}"
                )

            teacher_logprobs_l.append(
                pad_or_truncate_last_dim(batch[f'teacher_logprobs_{teacher_name}'], seqlen - 1, 0)
            )
            if has_ref_logprobs:
                ref_logprobs_l.append(
                    pad_or_truncate_last_dim(batch['ref_logprobs'], seqlen - 1, 0)
                )
            if has_rollout_logprobs:
                rollout_logprobs_l.append(
                    pad_or_truncate_last_dim(batch['rollout_log_probs'], seqlen - 1, 0)
                )
            if has_topk:
                stu_topk_logprobs_l.append(
                    pad_3d_seq_dim(batch['stu_topk_logprobs'], seqlen - 1, 0)
                )
                stu_topk_ids_l.append(pad_3d_seq_dim(batch['stu_topk_ids'], seqlen - 1, 0))
            sequence_lengths_l.append(batch['sequence_lengths'])

        tokens = torch.stack(tokens_l).cuda(non_blocking=non_blocking)
        advantages = torch.stack(advantages_l)
        mask = torch.stack(mask_l)
        logprobs = torch.stack(logprobs_l)
        teacher_logprobs = torch.stack(teacher_logprobs_l)
        ref_logprobs = torch.stack(ref_logprobs_l) if has_ref_logprobs else teacher_logprobs
        rollout_log_probs = torch.stack(rollout_logprobs_l) if has_rollout_logprobs else None
        sequence_lengths = torch.stack(sequence_lengths_l)
        stu_topk_logprobs = torch.stack(stu_topk_logprobs_l) if has_topk else None
        stu_topk_ids = torch.stack(stu_topk_ids_l) if has_topk else None

        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
            data=tokens,
            eod_token=0,  # unused
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            compute_attention_mask=False,
        )
        target = tokens.detach().clone()

        # mtp requires positon_ids and loss_mask
        mtp_labels, mtp_loss_mask = self._build_online_mtp_labels(
            tokens, mask, do_cp_split=not ppo_pack_seq
        )

        if dist.get_world_size(mpu.get_context_parallel_group()) > 1 and not ppo_pack_seq:
            tokens = get_tensor_on_this_cp_rank(tokens, 1, key_name="tokens")
            attention_mask = get_tensor_on_this_cp_rank(
                attention_mask, 2, key_name="attention_mask"
            )
            position_ids = get_tensor_on_this_cp_rank(position_ids, 1, key_name="position_ids")

        batch = {
            "tokens": tokens,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "advantages": advantages,
            "prev_log_probs": logprobs,
            "mask": mask,
            "ref_log_probs": ref_logprobs,
            "teacher_log_probs": teacher_logprobs,
            'target': target,
            'sequence_lengths': sequence_lengths,
            'mtp_labels': mtp_labels,
            'mtp_loss_mask': mtp_loss_mask,
        }
        if has_rollout_logprobs:
            batch["rollout_log_probs"] = rollout_log_probs
        if has_topk:
            batch["prev_topk_logprobs"] = stu_topk_logprobs
            batch["stu_topk_ids"] = stu_topk_ids

        required_keys = set()
        if mpu.get_pipeline_model_parallel_world_size() == 1:
            required_keys.update(batch.keys())
        else:
            required_keys.add("attention_mask")
            required_keys.add("sequence_lengths")
            required_keys.add("position_ids")
            if mpu.is_pipeline_first_stage():
                required_keys.update(("tokens", ))
            if mpu.is_pipeline_last_stage():
                required_keys.update(
                    (
                        "tokens",
                        "advantages",
                        "mask",
                        "prev_log_probs",
                        "ref_log_probs",
                        "teacher_log_probs",
                        "rollout_log_probs",
                        'target',
                        "prev_topk_logprobs",
                        "stu_topk_ids",
                    )
                )
                # mtp requires positon_ids and labels
                if self.config.training.online_mtp_sft:
                    required_keys.add("position_ids")
                    required_keys.add("mtp_labels")
                    required_keys.add("mtp_loss_mask")

        batch = {
            key:
                (
                    val.cuda(non_blocking=non_blocking)
                    if key in required_keys and val is not None else None
                )
            for key, val in batch.items()
        }

        fwd_kwargs = dict(
            input_ids=batch.pop("tokens"),
            position_ids=batch.pop("position_ids"),
            attention_mask=batch.pop("attention_mask"),
            labels=batch.get("mtp_labels", None),
            loss_mask=batch.get("mtp_loss_mask", None),
        )
        return batch, fwd_kwargs

    def sft_reroute_data_for_dynamic_cp(
        self,
        gbs_batches: List[Dict[str, Any]],
        pad_token_id: int,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[List[Dict[str, torch.Tensor]], int, float, float]:
        dp_group = mpu.get_data_parallel_group()
        tp_group = mpu.get_tensor_model_parallel_group()
        dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)

        dp_cp_size = dp_cp_group.size()
        tp_size = tp_group.size()

        dp_size = dp_group.size()
        cp_size = dp_cp_group.size() // dp_size
        total_dyn_cp_gpus = dp_cp_group.size()

        dist_config = self.config.policy.dist_config
        scheduler_type = dist_config.dynamic_cp_scheduler_type
        if scheduler_type == "smart_padding":
            dp_cp_pad = 2 * cp_size
        else:
            dp_cp_pad = 2 * dp_cp_size if dp_cp_size > 1 else 1
        tp_pad = tp_size if tp_size > 1 else 1
        pad_div = dp_cp_pad * tp_pad

        # 1. Shift labels and build clean sample dicts with the keys expected
        # by reroute_samples_to_dcp_ranks_by_keys / build_packed_microbatches.
        vocab_size = kwargs.get("vocab_size", 0)
        for i, batch in enumerate(gbs_batches):
            raw_len = batch["tokens"].shape[-1]
            pad_len = _round_up(raw_len, pad_div)
            tokens, labels, actual_len = self._prepare_tokens_and_labels(
                batch["tokens"],
                batch["labels"],
                pad_len,
                pad_token_id,
                vocab_size,
                pad_with_random_token,
            )
            gbs_batches[i] = dict(
                tokens=tokens,
                labels=labels,
                loss_mask=(labels != -100).to(torch.float32),
                position_ids=torch.arange(
                    tokens.shape[-1], dtype=torch.int64, device=tokens.device
                ),
                original_seq_len=torch.tensor([actual_len], dtype=torch.int32),
                padded_seq_len=torch.tensor([tokens.shape[-1]], dtype=torch.int32),
            )

        # 2. 根据调度器类型执行不同的调度和 packing 策略
        dev = torch.cuda.current_device()
        packed_keys = ["tokens", "labels", "loss_mask", "position_ids"]
        cat_keys = []

        if scheduler_type == "smart_padding":
            assert len(gbs_batches) % cp_size == 0, (
                f"gbs/dp_size ({len(gbs_batches)}) must be divisible by config_cp_size ({config_cp_size}). "
                f"Adjust gbs or context_parallel_size."
            )
            new_samples, num_micro_batches, seqlen_sum, seqlen_sq_sum = (
                sft_dyn_cp_schedule_smart_padding(
                    gbs_batches,
                    dp_group,
                    cp_size,
                    dist_config,
                    dev,
                    packed_keys,
                    cat_keys,
                )
            )
        else:
            assert scheduler_type == "default"
            global_id_seqlens_keys = [
                "tokens", "labels", "loss_mask", "position_ids", "original_seq_len",
                "padded_seq_len"
            ]
            dtype_map = {}
            new_samples, num_micro_batches, seqlen_sum, seqlen_sq_sum = (
                sft_dyn_cp_schedule_default(
                    gbs_batches,
                    dp_group,
                    tp_group,
                    dp_cp_group,
                    cp_size,
                    dp_size,
                    dist_config,
                    dev,
                    packed_keys,
                    cat_keys,
                    global_id_seqlens_keys,
                    dtype_map,
                )
            )

        return new_samples, num_micro_batches, seqlen_sum, seqlen_sq_sum


class OffPoilicyDistillPrepareDataForwardLLM(PrepareDataForwardLLM):
    @override
    def sft_train(
        self,
        batches: List[Dict[str, Any]],
        seq_len: int,
        pad_token_id: int,
        comput_attn_mask: bool = True,
        pad_with_random_token: bool = False,
        **kwargs,
    ):
        assert "input_teacher_logits" in kwargs, f"{kwargs=}"
        input_teacher_logits = kwargs["input_teacher_logits"]
        vocab_size = kwargs.get("vocab_size", 0)
        seq_len_shard_by_cp = seq_len // mpu.get_context_parallel_world_size()

        token_list = []
        label_list = []
        teacher_logits_list = []
        kl_alpha_mask = []
        for i, batch in enumerate(batches):
            assert batch["tokens"].ndim == 1, f"batch['tokens'].ndim={batch['tokens'].ndim}"
            teacher_logits = None
            if input_teacher_logits and mpu.is_pipeline_last_stage():
                teacher_logits = batch["teacher_logits"]
                assert seq_len % mpu.get_context_parallel_world_size(
                ) == 0, f"{seq_len=} {mpu.get_context_parallel_world_size()=}"
                assert teacher_logits.ndim == 2, f"teacher_logits.ndim={teacher_logits.ndim}"
                assert seq_len_shard_by_cp == teacher_logits.shape[
                    0], f"{seq_len_shard_by_cp=} != {teacher_logits.shape[0]}"
                # teacher 计算 logits 的时候就已经 pad 过了，而且是按照相同 tp 和 cp 拆分，所以不需要额外做 pad 或者cp 拆分这些
                # teacher_logits 本身是 pin_memory 的，所以直接 non_blocking = True 转 gpu 上速度最快
                teacher_logits_list.append(teacher_logits.cuda(non_blocking=True))

            token, label, _ = self._prepare_tokens_and_labels(
                batch["tokens"],
                batch["labels"],
                seq_len,
                pad_token_id,
                vocab_size,
                pad_with_random_token,
            )

            token_list.append(token)
            label_list.append(label)
            kl_alpha_mask.append(batch.get("offpd_loss_alpha", 0.0))

        tokens = torch.stack(token_list).view(len(token_list), -1).cuda(non_blocking=True)
        labels = torch.stack(label_list).view(len(token_list), -1).cuda(non_blocking=True)
        kl_alpha_mask = torch.tensor(kl_alpha_mask, device=tokens.device, dtype=torch.float32)
        batch_size = tokens.shape[0]
        teacher_logits = None
        if input_teacher_logits and mpu.is_pipeline_last_stage():
            teacher_logits = torch.stack(teacher_logits_list)
            teacher_logits = teacher_logits.view(batch_size, seq_len_shard_by_cp, -1)
            assert teacher_logits.ndim == 3, f"{teacher_logits.ndim=}"

        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            labels, -100, False, False, True, compute_attention_mask=comput_attn_mask
        )

        full_loss_mask = loss_mask
        if dist.get_world_size(mpu.get_context_parallel_group()) > 1:
            tokens, labels, loss_mask, position_ids, attention_mask = (
                self._sft_train_cp_chunk_data(
                    tokens, labels, loss_mask, position_ids, attention_mask
                )
            )

        batch = {
            "tokens": tokens,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "full_loss_mask": full_loss_mask,
        }

        # check all on gpu
        assert all([x.is_cuda for x in batch.values() if x is not None])
        fwd_kwargs = dict(
            input_ids=batch.pop("tokens"),
            position_ids=batch.pop("position_ids"),
            attention_mask=batch.pop("attention_mask"),
            labels=None,
        )
        # labels/loss_mask are already CP-split above; let the model compute the
        # MTP loss from them when online_mtp_sft is enabled.
        self._maybe_set_mtp_sft_fwd_kwargs(fwd_kwargs, batch["labels"], batch["loss_mask"])

        batch["kl_alpha_mask"] = kl_alpha_mask
        if input_teacher_logits:
            if mpu.is_pipeline_last_stage():
                assert teacher_logits.ndim == 3, f"teacher_logits.ndim={teacher_logits.ndim}"
                batch["teacher_logits"] = teacher_logits
            else:
                batch["teacher_logits"] = None
        return batch, fwd_kwargs


class DpoPrepareDataForwardLLM(PrepareDataForwardLLM):
    @override
    def sft_train(
        self,
        batches: List[Dict[str, Any]],
        seq_len: int,
        pad_token_id: int,
        comput_attn_mask: bool = True,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        ref_logps_l = []
        for batch in batches:
            assert "ref_logprobs" in batch, f"{batch=}"
            ref_logps = pad_or_truncate_last_dim(batch["ref_logprobs"], seq_len - 1, 0)
            ref_logps_l.append(ref_logps)

        non_blocking = True
        ref_logprobs = torch.stack(ref_logps_l).view(len(ref_logps_l),
                                                     -1).cuda(non_blocking=non_blocking)
        batch, fwd_kwargs = super(DpoPrepareDataForwardLLM, self).sft_train(
            batches, seq_len, pad_token_id, comput_attn_mask, pad_with_random_token, **kwargs
        )
        batch["ref_logprobs"] = ref_logprobs
        return batch, fwd_kwargs
