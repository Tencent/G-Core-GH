"""Pretrain and SFT GPT."""
import copy
import os
import token
import torch

from functools import partial
from typing import List, Optional, Union
from gpatch.core.utils import print_with_rank_and_datetime
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.training import get_args, get_timers, get_tokenizer, pretrain, print_rank_0
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core import tensor_parallel
from megatron.training.utils import get_ltor_masks_and_position_ids
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer.spec_utils import import_module
from megatron.core.utils import StragglerDetector
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron_datasets.tasks.math_rl_v3.sft_dataset import (
    update_epoch_and_line,
)
from dataclasses import dataclass

import megatron.legacy.model
from tasks.math_rl_v3 import args  # isort: skip

# NOTE: Loading `megatron.legacy.model` earlier fails due to circular import
from megatron_datasets.utils import print_rank_0, get_iterator

from tasks.gpt_oss.sft_dataset import GSftDataset

from gpatch.training.arguments import gpatch_extra_args
from gpatch.patch_mcore import init_gpatch_for_mcore
from gpatch.core.utils import split_data_cp_rank

from megatron_datasets.tasks.math_rl_v3.sft_dataset import (
    update_epoch_and_line,
    SftDataCollator,
)

try:
    from megatron.post_training.arguments import add_modelopt_args, modelopt_args_enabled
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt
    from megatron.post_training.model_provider import model_provider as model_provider_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False
from megatron_datasets.utils import random_pad_list

SPIKY_LOSS_FACTOR = 10

stimer = StragglerDetector()

# train_valid_test_datasets_provider = pretrain_gpt.train_valid_test_datasets_provider


def model_provider(
    pre_process=True,
    post_process=True,
    vp_stage: Optional[int] = None
) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()

    if has_nvidia_modelopt and modelopt_args_enabled(args):  # [ModelOpt]
        return model_provider_modelopt(pre_process, post_process)

    use_te = args.transformer_impl == "transformer_engine"

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(
            True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,
            # record stack information for the trace events
            trace_alloc_record_context=True,
        )

        def oom_observer(device, alloc, device_alloc, device_free):
            # snapshot right after an OOM happened
            print('saving allocated state during OOM')
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump

            dump(
                snapshot,
                open(f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}", 'wb'),
            )

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    # TODO(astrachang): 这东西怎么用yaml传tuple的啊
    if args.model_arch == "gpt_oss_moe":
        config.window_size = (128, 0)
        config.window_size = tuple(config.window_size)
        # config.window_attn_skip_freq = [1, 0] * (config.num_layers // 2)
        config.bias_dropout_fusion = False

        # NV的Config里面还没yarn的参数呢，要手动设置
        config.position_embedding_type = args.position_embedding_type
        config.yarn_rotary_scaling_factor = args.yarn_rotary_scaling_factor
        config.yarn_original_max_position_embeddings = args.yarn_original_max_position_embeddings
        config.yarn_beta_fast = args.yarn_beta_fast
        config.yarn_beta_slow = args.yarn_beta_slow
        config.yarn_mscale = args.yarn_mscale
        config.yarn_mscale_all_dim = args.yarn_mscale_all_dim
        config.yarn_correction_range_round_to_int = args.yarn_correction_range_round_to_int

    print_rank_0(f'{config.recompute_method=}')
    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else:  # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts:
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(
                    config,
                    use_transformer_engine=use_te,
                    normalization=args.normalization,
                    qk_l2_norm=args.qk_l2_norm,
                    vp_stage=vp_stage
                )
            elif args.heterogeneous_layers_config_path is not None:
                transformer_layer_spec = get_gpt_heterogeneous_layer_spec(config, use_te)
            else:
                # Define the decoder layer spec
                transformer_layer_spec = _get_transformer_layer_spec(use_te, config)
        mtp_block_spec = None
        if args.mtp_num_layers is not None:
            if hasattr(transformer_layer_spec,
                       'layer_specs') and len(transformer_layer_spec.layer_specs) == 0:
                # Get the decoder layer spec explicitly if no decoder layer in the last stage,
                # Only happens with block spec (TransformerBlockSubmodules) when using MoE.
                transformer_layer_spec_for_mtp = _get_transformer_layer_spec(use_te, config)
            else:
                transformer_layer_spec_for_mtp = transformer_layer_spec
            mtp_block_spec = get_gpt_mtp_block_spec(
                config,
                transformer_layer_spec_for_mtp,
                use_transformer_engine=use_te,
                vp_stage=vp_stage
            )

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
        )

    return model


def get_batch(data_iterator):
    """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
    args = get_args()
    # Items and their type.
    keys = ['input_ids', 'labels']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    if args.px_use_indexed_jsonl_dataset:
        update_epoch_and_line(
            args.train_data_consuming_progresses, torch.distributed.get_rank(), data
        )
    # print_with_rank_and_datetime(f"[get_batch] 0 tokens_ {data['input_ids'].shape} labels_ {data['labels'].shape}")
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['input_ids'].long()
    labels_ = data_b['labels'].long()
    # print_with_rank_and_datetime(f"[get_batch] 1 tokens_ {tokens_.shape} labels_ {labels_.shape}")
    if args.px_use_indexed_jsonl_dataset:
        labels = labels_.contiguous()
        tokens = tokens_.contiguous()
    else:
        raise NotImplementedError("somgthing wrong")
    # print_with_rank_and_datetime(f"[get_batch] 2 tokens_ {tokens_.shape} labels_ {labels_.shape}")

    # pad 我们自己处理了
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        labels, -100, 0, args.reset_position_ids, args.reset_attention_mask, args.eod_mask_loss,
        False
    )

    if args.context_parallel_size > 1:
        tokens = split_data_cp_rank(tokens, mpu.get_context_parallel_world_size(), 1)
        labels = split_data_cp_rank(labels, mpu.get_context_parallel_world_size(), 1)
        loss_mask = split_data_cp_rank(loss_mask, mpu.get_context_parallel_world_size(), 1)
        attention_mask = split_data_cp_rank(
            attention_mask, mpu.get_context_parallel_world_size(), 2
        )
        position_ids = split_data_cp_rank(position_ids, mpu.get_context_parallel_world_size(), 1)

    if os.environ.get("PX_DEBUG_TRAIN_LOG", "0") == "1":
        tokens_non_pad_lengths = (tokens != get_tokenizer()._tokenizer.pad_token_id).sum(dim=1)
        labels_non_pad_lengths = (labels != -100).sum(dim=1)
        first_non_pad_indices = (labels != -100).max(dim=1).indices
        tokens_sum = [v[:tokens_non_pad_lengths[i]].sum() for i, v in enumerate(tokens)]
        labels_sum = [
            v[first_non_pad_indices[i]:first_non_pad_indices[i] + labels_non_pad_lengths[i]].sum()
            for i, v in enumerate(labels)
        ]
        real_tokens_ = [tokens[i][:tokens_non_pad_lengths[i]] for i in range(tokens.shape[0])]
        real_labels_ = [
            v[first_non_pad_indices[i]:first_non_pad_indices[i] + labels_non_pad_lengths[i]]
            for i, v in enumerate(labels)
        ]
        print(
            f"trace input {torch.distributed.get_rank()} {tokens_non_pad_lengths=} {first_non_pad_indices=} "
            f"{labels_non_pad_lengths=} sum {tokens_sum} {labels_sum} {tokens.shape}"
        )

    return tokens, labels, loss_mask, None, position_ids


def loss_func(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[GPTModel] = None
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (GPTModel, optional): The model (can be wrapped)

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    if has_nvidia_modelopt and modelopt_args_enabled(args):  # [ModelOpt]
        return loss_func_modelopt(loss_mask, output_tensor, model=model)

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)

    # mcore 不在这里做 cp 的 allreduce, 因为 megatron 把 allreduce 环节放到了 train_step 外面
    # 但是如果训练数据里 pad token 太多，某一个 cp 全是 pad 出来，出来的 loss 为 0.0，num_tokens 也为 0.0
    # 这样算出来的梯度会变成 nan，导致检查不过
    total_tokens = loss_mask.sum()
    loss = torch.cat([loss.view(1), total_tokens.view(1)])

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(
            loss, group=mpu.get_context_parallel_group(), op=torch.distributed.ReduceOp.AVG
        )

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    reporting_loss = loss.clone().detach()
    local_num_tokens = loss[1].sum().clone().detach().to(torch.int)
    return (
        loss[0].clone(),
        local_num_tokens,
        {
            'lm loss': reporting_loss
        },
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()
    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
        # if torch.distributed.get_rank() == 0:
        #     folder = 'debug-tmp'
        #     os.makedirs(folder, exist_ok=True)
        #     torch.save({"tokens": tokens, "labels": labels, "loss_mask": loss_mask, "attention_mask": attention_mask, "position_ids": position_ids}, f"{folder}/inputs-{torch.distributed.get_rank()}.pt")
        #     import sys; sys.exit()
        # d = torch.load(f"debug-tmp/inputs-0.pt", weights_only=False)
        # tokens = d['tokens'].cuda()
        # labels = d['labels'].cuda()
        # loss_mask = d['loss_mask'].cuda()
        # attention_mask = None
        # position_ids = d['position_ids'].cuda()
        # print_with_rank_and_datetime(f"tokens: {tokens.device=} {tokens.shape=} {labels.shape=} {loss_mask.shape=} {position_ids.shape=}")
    timers('batch-generator').stop()
    with stimer:
        output_tensor = model(
            tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
        )
    # import sys; sys.exit()
    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model)


@dataclass
class SftDataCollator:
    hf_tokenizer: AutoTokenizer
    seq_len: int

    def __call__(self, batch):
        pad_token_id = self.hf_tokenizer.pad_token_id
        input_ids_l = []
        labels_l = []
        for item in batch:
            input_ids = item['input_ids']
            labels = item['labels']
            tmp_len = self.seq_len + 1
            if len(input_ids) < tmp_len:
                to_pad = tmp_len - len(input_ids)
                input_ids = random_pad_list(input_ids, to_pad)
                labels += [-100] * to_pad
            else:
                input_ids = input_ids[-tmp_len:]
                labels = labels[-tmp_len:]

            input_ids = input_ids[:-1]
            labels = labels[1:]
            input_ids_l.append(input_ids)
            labels_l.append(labels)

        input_ids = torch.as_tensor(input_ids_l, dtype=torch.int64)
        labels = torch.as_tensor(labels_l, dtype=torch.int64)
        ret = {
            'input_ids': input_ids,
            'labels': labels,
            'train': torch.as_tensor([item['train'] for item in batch], dtype=torch.bool),
            'epoch': torch.as_tensor([item['epoch'] for item in batch], dtype=torch.int64),
            'line': torch.as_tensor([item['line'] for item in batch], dtype=torch.int64),
        }
        return ret


def gdataset_map_fn(tokenizer, example):
    # 例子：
    # rank=1 dp_rank=0 sp_rank=1 {
    # 'question': 'James buys 3 dirt bikes for $150 each and 4 off-...  How much did he pay for everything?',
    # 'answer': 'The dirtbikes cost 3*150=$<<3*150=450>>450\nThe ...\n#### 1825'
    # }

    # prompt
    chat = [
        {
            'role': 'user',
            'content': example['user'],
        },
    ]
    prompt_input_ids = tokenizer._tokenizer.apply_chat_template(
        chat,
        add_special_tokens=False,
        tokenize=True,
        add_generation_prompt=True,
    )

    # prompt + answer
    chat = [
        {
            'role': 'user',
            'content': example['user'],
        },
        {
            'role': 'assistant',
            'content': example['final'],
        },
    ]
    text_input_ids = tokenizer._tokenizer.apply_chat_template(
        chat,
        add_special_tokens=False,
        tokenize=True,
        add_generation_prompt=False,
    )
    assert text_input_ids[:len(prompt_input_ids)] == prompt_input_ids

    # label
    labels = copy.deepcopy(text_input_ids)
    labels[:len(prompt_input_ids)] = [-100] * len(prompt_input_ids)

    return {
        'input_ids': text_input_ids,
        'labels': labels,
    }


def train_valid_test_datasets_provider(train_val_test_num_samples=None):
    tokenizer = get_tokenizer()
    args = get_args()
    train_dataset = GSftDataset(
        tokenizer=tokenizer,
        seq_len=args.seq_length,
        metadata_file=args.px_data_config_path,
        dp_rank=mpu.get_data_parallel_rank(),
        dp_size=mpu.get_data_parallel_world_size(),
        gbs=args.global_batch_size,
        mbs=args.micro_batch_size,
        shuffling_buffer_size=args.px_shuffle_buffer_size,
        train=True,
        seed=args.seed,
        dataset_map_fn=gdataset_map_fn,
    )
    sft_data_collator = SftDataCollator(hf_tokenizer=tokenizer._tokenizer, seq_len=args.seq_length)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
        prefetch_factor=8,
        drop_last=True,
        pin_memory=True,
        collate_fn=sft_data_collator,
    )

    return get_iterator(train_dataloader), None, None


if __name__ == "__main__":
    init_gpatch_for_mcore()
    # Temporary for transition to core datasetss
    train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'HuggingFaceTokenizer'},
        extra_args_provider=gpatch_extra_args,
    )
