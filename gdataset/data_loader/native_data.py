from typing import Any, Literal, Mapping, Optional, Union

import torch

from gdataset import GDatasetV4
from gdataset.data_loader.llamafactory_bridge import GDatasetV4Adaptor
from gdataset.data_loader.processor import ProcessorWrapper, StackedMap, _get_dataset_processor
from gdataset.feat import PilImageListFeat

from gpatch.training.v3.ppo_actor import iter_to_ppo_epoch_step


def sort_by_prompt_len(sample):
    return sample["tokenizer_len"]


def build_gcore_datasets(args, feats, llamafactory_config, dp_size, dp_rank):
    use_for_hf = False
    stage = "sft" if not args.use_grpo else "ppo"
    processor_args = (
        llamafactory_config.data_args, stage, llamafactory_config.template,
        llamafactory_config.tokenizer_module["tokenizer"],
        llamafactory_config.tokenizer_module["processor"]
    )
    train_path_like = args.gdatasetv4_train_metadata_file
    eval_path_like = args.gdatasetv4_eval_metadata_file
    mask_history = args.mask_history
    use_grpo = args.use_grpo
    if use_grpo:
        assert mask_history, f"mask_history must be True when use grpo"

    gbs = args.global_batch_size
    consumed = args.iteration * gbs
    epoch = 0
    if args.use_grpo and consumed > 0:
        gbs = args.ppo_rollout_global_batch_size
        epoch, ppo_step = iter_to_ppo_epoch_step(args.iteration)
        consumed = ppo_step * gbs

    smart_padding_compare_func = None
    smart_padding_buffer_size = 0
    if args.px_inputs_pad_to_longest:
        smart_padding_compare_func = sort_by_prompt_len
        smart_padding_buffer_size = args.px_smart_padding_buffer_size

    train_ds = GDatasetV4(
        train_path_like,
        dp_rank=dp_rank,
        dp_size=dp_size,
        gbs=gbs,
        shuffling_buffer_size=args.px_shuffle_buffer_size,
        consumed=consumed,
        feats=feats,
        seed=42,
        smart_padding_compare_func=smart_padding_compare_func,
        smart_padding_buffer_size=smart_padding_buffer_size,
        mbs=args.micro_batch_size,
    )

    adaptor_map_func = GDatasetV4Adaptor(None, llamafactory_config.data_args)
    train_map_fn = ProcessorWrapper(_get_dataset_processor(*processor_args))
    stack_map = StackedMap()
    stack_map.add(adaptor_map_func)
    stack_map.add(train_map_fn)
    train_ds.map(stack_map)
    train_ds.set_epoch(epoch)

    eval_ds = None
    if eval_path_like is not None:
        eval_gbs = gbs
        if args.use_grpo:
            eval_gbs = args.ppo_eval_rollout_global_batch_size
        eval_ds = GDatasetV4(
            eval_path_like,
            dp_rank=dp_rank,
            dp_size=dp_size,
            gbs=eval_gbs,
            shuffling_buffer_size=args.px_shuffle_buffer_size,
            consumed=0,
            feats=feats,
            seed=42,
        )
        stack_map = StackedMap()
        stack_map.add(adaptor_map_func)
        eval_map_fn = ProcessorWrapper(_get_dataset_processor(*processor_args))
        stack_map.add(eval_map_fn)
        eval_ds.map(stack_map)
        eval_ds.set_epoch(0)
    test_ds = None

    return train_ds, eval_ds, test_ds


def build_gcore_v4_dataloaders(
    args, llamafactory_config, collator, rank, dp_size, dp_rank, feats=None
):
    if feats is None:
        feats = {
            'images':
                PilImageListFeat(
                    lmdb=True,
                    return_src_data=True,
                    convert_to_rgb=True,
                    new_name="__images_feat__",
                ),
        }
    train_ds, eval_ds, test_ds = build_gcore_datasets(
        args, feats, llamafactory_config, dp_size, dp_rank
    )
    collate_func = collator

    batch_size = args.micro_batch_size
    if args.use_grpo:
        batch_size = args.ppo_rollout_micro_batch_size
    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_func,
        prefetch_factor=args.px_dataloader_prefetch_factor,
    )

    eval_dataloader = None
    if eval_ds is not None:
        eval_batch_size = batch_size
        if args.use_grpo:
            eval_batch_size = args.ppo_eval_rollout_micro_batch_size
        eval_dataloader = torch.utils.data.DataLoader(
            eval_ds,
            batch_size=eval_batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_func,
            prefetch_factor=args.px_dataloader_prefetch_factor,
        )
    test_dataloader = None
    if test_ds is not None:
        test_dataloader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_func,
            prefetch_factor=args.px_dataloader_prefetch_factor,
        )

    return train_dataloader, eval_dataloader, test_dataloader
