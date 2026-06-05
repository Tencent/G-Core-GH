from abc import ABC, abstractmethod
from types import SimpleNamespace

import torch
from typing_extensions import override

from gdataset.data_loader.collator import PPODataCollator, SFTDataCollator
from gdataset.data_loader.constants import IGNORE_INDEX
from gdataset.data_loader.llamafactory_bridge import init_llamafactory
from gdataset.data_loader.llamafactory_data import build_llama_fc_dataloaders
from gdataset.data_loader.native_data import build_gcore_v4_dataloaders
from gdataset.data_loader.rope_index import get_index_helper
from megatron_datasets.utils import get_iterator

try:
    from gdataset.data_loader.energon_data import build_energon_data_loaders
except:
    pass


class BaseDataItersBuilder(ABC):
    def __init__(self, args):
        self._args = args
        self._llamafactory_config = init_llamafactory(args)

    @property
    def llamafactory_config(self):
        return self._llamafactory_config

    @abstractmethod
    def build_collator(self, rope_index_func_provider=None):
        ...

    def _build_llamafactory_data_iters(self, collator, rank, dp_size, dp_rank):
        train_dl, eval_dl, test_dl = build_llama_fc_dataloaders(
            self._args, self.llamafactory_config, collator, rank, dp_size, dp_rank
        )

        return get_iterator(train_dl), get_iterator(eval_dl), get_iterator(test_dl)

    def _build_gcore_v4_data_iters(self, collator, rank, dp_size, dp_rank):
        train_dl, eval_dl, test_dl = build_gcore_v4_dataloaders(
            self._args, self.llamafactory_config, collator, rank, dp_size, dp_rank
        )
        return get_iterator(train_dl), get_iterator(eval_dl), get_iterator(test_dl)

    def _build_energon_data_iters(self, collator, rank, dp_size, dp_rank):
        train_dl, eval_dl, test_dl = build_energon_data_loaders(
            self._args, self.llamafactory_config, collator, rank, dp_size, dp_rank
        )
        return train_dl, eval_dl, test_dl

    def build_data_iters(self, collator, rank, dp_size, dp_rank):
        if self._args.dataset_impl == "llamafactory":
            return self._build_llamafactory_data_iters(collator, rank, dp_size, dp_rank)
        elif self._args.dataset_impl == "v4_dataloaders":
            return self._build_gcore_v4_data_iters(collator, rank, dp_size, dp_rank)
        elif self._args.dataset_impl == "energon":
            return self._build_energon_data_iters(collator, rank, dp_size, dp_rank)
        else:
            assert False, f"unsupported dataset_impl {self._args.dataset_impl}"

    def build_dataloaders(self, rank, dp_size, dp_rank):
        collator = self.build_collator()
        if self._args.dataset_impl == "llamafactory":
            return build_llama_fc_dataloaders(
                self._args,
                self.llamafactory_config,
                collator,
                rank,
                dp_size,
                dp_rank,
            )
        elif self._args.dataset_impl == "v4_dataloaders":
            return build_gcore_v4_dataloaders(
                self._args,
                self.llamafactory_config,
                collator,
                rank,
                dp_size,
                dp_rank,
            )
        elif self._args.dataset_impl == "energon":
            return build_energon_data_loaders(
                self._args,
                self.llamafactory_config,
                collator,
                rank,
                dp_size,
                dp_rank,
            )
        else:
            assert False, f"unsupported dataset_impl {self._args.dataset_impl}"

    def build(self, rank=0, dp_size=1, dp_rank=0):
        collator = self.build_collator()
        train_iters, valid_iters, test_iters = self.build_data_iters(
            collator, rank, dp_size, dp_rank
        )
        return train_iters, valid_iters, test_iters


class BaseRopeIndexFunc(ABC):
    @abstractmethod
    def __call__(self, features, mm_inputs, origin_json_data_list):
        ...


class Glm4VGetRopeIndexFunc(BaseRopeIndexFunc):
    def __init__(self, args, llamafactory_config):
        self._args = args
        self._llamafactory_config = llamafactory_config
        config_path = llamafactory_config.model_args.model_name_or_path
        self.index_helper = get_index_helper("glm4v", config_path)

    @override
    def __call__(self, features, mm_inputs, origin_json_data_list):
        input_ids = features.get("input_ids", None)
        image_grid_thw = mm_inputs.get("image_grid_thw", None)
        video_grid_thw = mm_inputs.get("video_grid_thw", None)
        second_per_grid_ts = mm_inputs.get("second_per_grid_ts", None)
        """Build masks and position id for left to right model."""
        # Position ids. [3 X bs X seqlen]
        assert input_ids is not None
        assert image_grid_thw is not None
        assert video_grid_thw is None
        assert second_per_grid_ts is None
        position_ids, _ = self.index_helper.get_rope_index(input_ids, image_grid_thw)
        return {"position_ids": position_ids}


class Qwen3VGetRopeIndexFunc(BaseRopeIndexFunc):
    def __init__(self, args, llamafactory_config):
        self._args = args
        self._llamafactory_config = llamafactory_config
        config_path = llamafactory_config.model_args.model_name_or_path
        self.index_helper = get_index_helper("qwen3_vl", config_path)

    @override
    def __call__(self, features, mm_inputs, origin_json_data_list):
        input_ids = features.get("input_ids", None)
        image_grid_thw = mm_inputs.get("image_grid_thw", None)
        video_grid_thw = mm_inputs.get("video_grid_thw", None)
        second_per_grid_ts = mm_inputs.get("second_per_grid_ts", None)
        """Build masks and position id for left to right model."""
        # Position ids. [3 X bs X seqlen]
        assert input_ids is not None
        assert image_grid_thw is not None
        assert video_grid_thw is None
        assert second_per_grid_ts is None
        position_ids, _ = self.index_helper.get_rope_index(input_ids, image_grid_thw)
        return {"position_ids": position_ids}


class Qwen3_5VGetRopeIndexFunc(BaseRopeIndexFunc):
    def __init__(self, args, llamafactory_config):
        self._args = args
        self._llamafactory_config = llamafactory_config
        config_path = llamafactory_config.model_args.model_name_or_path
        self.index_helper = get_index_helper("qwen3_5", config_path)

    @override
    def __call__(self, features, mm_inputs, origin_json_data_list):
        input_ids = features.get("input_ids", None)
        image_grid_thw = mm_inputs.get("image_grid_thw", None)
        video_grid_thw = mm_inputs.get("video_grid_thw", None)
        second_per_grid_ts = mm_inputs.get("second_per_grid_ts", None)
        """Build masks and position id for left to right model."""
        # Position ids. [3 X bs X seqlen]
        assert input_ids is not None
        assert image_grid_thw is not None
        assert video_grid_thw is None
        assert second_per_grid_ts is None
        position_ids, _ = self.index_helper.get_rope_index(input_ids, image_grid_thw)
        return {"position_ids": position_ids}


class DefaultDataItersBuilder(BaseDataItersBuilder):
    def rope_index_func_provider(self):
        # TODO(hessianliu): register a GetRopeIndexFunc for all supported model
        # TODO(hessianliu): model_arch is deprecated in trainer v4
        if self._args.model_arch in ["glm4v_moe", "glm4v"]:
            return Glm4VGetRopeIndexFunc(self._args, self._llamafactory_config)
        if self._args.model_arch in ["qwen3_vl", "qwen3_vl_moe"]:
            return Qwen3VGetRopeIndexFunc(self._args, self._llamafactory_config)
        if self._args.model_arch in ["qwen3_5", "qwen3_5_moe"]:
            return Qwen3_5VGetRopeIndexFunc(self._args, self._llamafactory_config)
        assert False, f"{self._args.model_arch}"
        return None

    @override
    def build_collator(self, rope_index_func_provider=None):

        rope_index_func_provider = rope_index_func_provider or self.rope_index_func_provider

        rope_index_func = rope_index_func_provider()

        llamafactory_config = self.llamafactory_config
        training_args = llamafactory_config.training_args
        tokenizer_module = llamafactory_config.tokenizer_module
        template = llamafactory_config.template
        data_args = llamafactory_config.data_args
        tokenizer = tokenizer_module["tokenizer"]

        collator_cls = PPODataCollator if self._args.use_grpo else SFTDataCollator
        padding = "max_length"

        data_collator = collator_cls(
            template=template,
            model=None,
            padding=padding,
            max_length=self._args.seq_length,
            pad_to_multiple_of=8 if training_args.do_train else None,
            label_pad_token_id=IGNORE_INDEX
            if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
            post_collate_hook=rope_index_func,
            **tokenizer_module,
        )
        return data_collator
