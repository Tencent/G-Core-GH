import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import TestCase, mock

import torch


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _package(name: str, **attributes: object) -> ModuleType:
    module = _module(name, **attributes)
    module.__path__ = []
    return module


def _pad_or_truncate_last_dim(
    tensor: torch.Tensor,
    length: int,
    value: object,
    pad_with_random_token: bool = False,
    vocab_size: int = 0,
    forbidden_token_ids: object = None,
) -> torch.Tensor:
    assert not pad_with_random_token
    if tensor.shape[-1] < length:
        tensor = torch.nn.functional.pad(tensor, (0, length - tensor.shape[-1]), value=value)
    if tensor.shape[-1] > length:
        tensor = tensor[..., :length]
    return tensor


def _load_qwenvl_dataset_map() -> ModuleType:
    stubs = {
        "gdataset":
            _package("gdataset", GDatasetV4=object),
        "gdataset.data_loader":
            _package("gdataset.data_loader"),
        "gdataset.data_loader.rope_index":
            _module(
                "gdataset.data_loader.rope_index",
                get_index_helper=lambda *args, **kwargs: None,
            ),
        "gdataset.feat":
            _module("gdataset.feat", PilImageListFeat=object),
        "megatron_datasets.mega_indexed_jsonl_dataset_mm":
            _module(
                "megatron_datasets.mega_indexed_jsonl_dataset_mm",
                MegaIndexedJsonlDatasetMM=object,
            ),
        "megatron_datasets.mm_dataset":
            _module(
                "megatron_datasets.mm_dataset",
                MultiModalDatasetMap=object,
                convert_conversations=lambda *args, **kwargs: None,
                refact_conversations=lambda *args, **kwargs: None,
            ),
        "megatron_datasets.tools":
            _package("megatron_datasets.tools"),
        "megatron_datasets.tools.lmdb_read_cli":
            _module(
                "megatron_datasets.tools.lmdb_read_cli",
                fetch_images_from_lmdb=lambda *args: None,
            ),
        "megatron_datasets.utils":
            _module(
                "megatron_datasets.utils",
                build_forbidden_token_ids=lambda *args, **kwargs: set(),
                get_iterator=lambda value: value,
                random_pad_list=lambda value, *args: value,
            ),
        "transformers":
            _package(
                "transformers",
                AutoConfig=object,
                Qwen2VLImageProcessorFast=object,
                Qwen3VLVideoProcessor=object,
                WhisperFeatureExtractor=object,
                Qwen3VLProcessor=object,
            ),
        "transformers.image_utils":
            _module(
                "transformers.image_utils",
                is_valid_image=lambda *args, **kwargs: False,
            ),
        "transformers.models":
            _package("transformers.models"),
        "transformers.models.auto":
            _package("transformers.models.auto"),
        "transformers.models.auto.processing_auto":
            _module(
                "transformers.models.auto.processing_auto",
                AutoProcessor=object,
            ),
        "transformers.utils":
            _package("transformers.utils"),
        "transformers.utils.import_utils":
            _module(
                "transformers.utils.import_utils",
                is_torchcodec_available=lambda: False,
            ),
        "transformers.video_utils":
            _module(
                "transformers.video_utils",
                VideoMetadata=object,
            ),
        "gpatch_v4":
            _package("gpatch_v4"),
        "gpatch_v4.utils":
            _package("gpatch_v4.utils"),
        "gpatch_v4.utils.training_utils":
            _module(
                "gpatch_v4.utils.training_utils",
                pad_or_truncate_last_dim=_pad_or_truncate_last_dim,
            ),
        "mbridge":
            _package("mbridge"),
        "mbridge.core":
            _package("mbridge.core"),
        "mbridge.core.util":
            _module(
                "mbridge.core.util",
                expand_thw=lambda value: value,
                qwen2vl_pad_and_split=lambda *args: None,
            ),
        "mbridge.models":
            _package("mbridge.models"),
        "mbridge.models.qwen3_vl":
            _package("mbridge.models.qwen3_vl"),
        "mbridge.models.qwen3_vl.utils":
            _module(
                "mbridge.models.qwen3_vl.utils",
                reorganize_inputs=lambda **kwargs: (None, None, None),
            ),
    }
    path = Path(__file__).parents[2] / "megatron_datasets/qwenvl_dataset_map.py"
    spec = importlib.util.spec_from_file_location("qwenvl_dataset_map_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    original_modules = {}
    missing_modules = []
    for name in stubs:
        if name in sys.modules:
            original_modules[name] = sys.modules[name]
        else:
            missing_modules.append(name)
    sys.modules.update(stubs)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.update(original_modules)
        for name in missing_modules:
            if name in sys.modules:
                del sys.modules[name]
    return module


def _make_instance(pixel_values: torch.Tensor | None) -> dict[str, object]:
    input_ids = torch.tensor([1, 5, 6, 0], dtype=torch.int64)
    image_grid_thw = None
    if pixel_values is not None:
        image_grid_thw = torch.tensor([[1, 1, 1]], dtype=torch.int64)
    image_input_mask = input_ids == 1
    if pixel_values is None:
        image_input_mask = torch.zeros(4, dtype=torch.bool)
    return {
        "input_ids": input_ids,
        "labels": torch.tensor([-100, 5, 6, 0], dtype=torch.int64),
        "attention_mask": torch.ones(4, dtype=torch.bool),
        "prompt_len": torch.tensor(1, dtype=torch.int64),
        "tokenizer_len": torch.tensor(4, dtype=torch.int64),
        "sequence_lengths": torch.tensor(4, dtype=torch.int64),
        "meta_info": None,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "image_input_mask": image_input_mask,
        "pixel_values_videos": None,
        "video_grid_thw": None,
        "video_input_mask": torch.zeros(4, dtype=torch.bool),
    }


class TestTrainerV4DataCollatorForQwenVl(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_qwenvl_dataset_map()

    def test_mixed_batch_preserves_vision_sample_positions(self) -> None:
        module = self.module
        collator = module.TrainerV4DataCollatorForQwenVl.__new__(
            module.TrainerV4DataCollatorForQwenVl
        )
        collator.is_dpo = False
        collator.use_grpo = False
        collator.model_arch = "qwen3_vl"
        collator.only_return_last_hidden_state = False
        collator.pad_token_id = 0
        collator.mrope_index = mock.Mock(
            hf_class=None,
            config=SimpleNamespace(
                image_token_id=1,
                video_token_id=2,
                vision_config=SimpleNamespace(spatial_merge_size=1),
            ),
        )
        collator.mrope_index.get_rope_index.return_value = (
            torch.zeros((3, 1, 4), dtype=torch.int64),
            None,
        )

        image_a = torch.ones((1, 3), dtype=torch.float32)
        image_b = torch.full((1, 3), 2.0, dtype=torch.float32)
        instances = [
            _make_instance(None),
            _make_instance(image_a),
            _make_instance(None),
            _make_instance(image_b),
        ]

        def reorganize_inputs(**kwargs: object) -> tuple[object, object, torch.Tensor]:
            pixel_values = kwargs["pixel_values"]
            image_input_mask = kwargs["image_input_mask"]
            assert isinstance(image_input_mask, torch.Tensor)
            if pixel_values is None:
                return None, None, image_input_mask
            assert isinstance(pixel_values, torch.Tensor)
            image_grid_thw = kwargs["image_grid_thw"]
            assert isinstance(image_grid_thw, torch.Tensor)
            return pixel_values, image_grid_thw, image_input_mask

        with mock.patch.object(module, "reorganize_inputs", side_effect=reorganize_inputs):
            result = collator(instances)

        self.assertEqual(len(result["vision_data"]), 4)
        self.assertEqual(len(result["vision_grid_thw"]), 4)
        self.assertEqual(
            [value is None for value in result["vision_data"]], [True, False, True, False]
        )
        self.assertIs(result["vision_data"][1], image_a)
        self.assertIs(result["vision_data"][3], image_b)
        self.assertIsNone(result["vision_grid_thw"][0])
        self.assertIsNone(result["vision_grid_thw"][2])

    def test_packed_collator_reorganizes_vision_inputs(self) -> None:
        module = self.module
        collator = module.TrainerV4DataCollatorForQwenVlPacked.__new__(
            module.TrainerV4DataCollatorForQwenVlPacked
        )
        collator.pad_token_id = 0
        collator.vocab_size = 16
        collator.pad_with_random_token = False
        collator.forbidden_token_ids = []
        collator.model_arch = "qwen3_vl"
        collator.mrope_index = mock.Mock(
            hf_class=None,
            config=SimpleNamespace(
                image_token_id=1,
                video_token_id=2,
                vision_config=SimpleNamespace(spatial_merge_size=1),
            ),
        )
        collator.mrope_index.get_rope_index.return_value = (
            torch.zeros((3, 1, 4), dtype=torch.int64),
            None,
        )

        text_item = _make_instance(None)
        text_item["input_ids"] = torch.tensor([3, 5, 6, 0], dtype=torch.int64)
        image_item = _make_instance(
            torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        )

        def reorganize_inputs(**kwargs: object):
            image_mask = kwargs["image_input_mask"]
            video_mask = kwargs["video_input_mask"]
            assert isinstance(image_mask, torch.Tensor)
            assert isinstance(video_mask, torch.Tensor)
            return (
                kwargs["pixel_values"],
                kwargs["image_grid_thw"],
                image_mask | video_mask,
            )

        with (
            mock.patch.object(
                module,
                "reorganize_inputs",
                side_effect=reorganize_inputs,
            ) as reorganize_mock,
            mock.patch.object(
                module,
                "expand_thw",
                side_effect=lambda value: value,
            ) as expand_mock,
        ):
            result = collator(
                [text_item, image_item],
                align=4,
                pack_bin_size=8,
            )

        reorganize_mock.assert_called_once()
        expand_mock.assert_called_once()
        self.assertEqual(result["tokens"].shape, (8,))
        self.assertEqual(result["image_input_mask"].shape, (1, 8))
        self.assertEqual(result["vision_data"].shape, (1, 3))
        self.assertEqual(result["vision_grid_thw"].shape, (1, 3))
        for key in (
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "video_input_mask",
        ):
            self.assertNotIn(key, result)

    def test_omni_rope_index_kwargs(self) -> None:
        # Omni 无 attention_mask=None 兜底；mask=None 时 audio_seqlens 也是 None
        module = self.module
        input_ids = torch.ones((1, 4), dtype=torch.int64)
        attention_mask, kwargs = module.get_omni_rope_index_kwargs(
            {
                "audio_feature_lengths": torch.tensor([8], dtype=torch.int64),
                "video_second_per_grid": None,
            },
            input_ids,
        )
        self.assertEqual(tuple(attention_mask.shape), (1, 4))
        self.assertTrue(torch.all(attention_mask == 1))
        self.assertEqual(int(kwargs["audio_seqlens"].item()), 8)
        self.assertIsNone(kwargs["second_per_grids"])

        _, kwargs = module.get_omni_rope_index_kwargs(
            {
                "audio_feature_lengths": None,
                "video_second_per_grid": None,
            },
            input_ids,
        )
        self.assertIsNone(kwargs["audio_seqlens"])

    def test_packed_omni_passes_ones_attention_mask(self) -> None:
        # qwen3_omni get_rope_index 没有 attention_mask=None 兜底
        module = self.module
        collator = module.TrainerV4DataCollatorForQwenVlPacked.__new__(
            module.TrainerV4DataCollatorForQwenVlPacked
        )
        collator.pad_token_id = 0
        collator.vocab_size = 16
        collator.pad_with_random_token = False
        collator.forbidden_token_ids = []
        collator.model_arch = "qwen3_omni_moe"
        collator.mrope_index = mock.Mock(
            hf_class=None,
            config=SimpleNamespace(
                image_token_id=1,
                video_token_id=2,
                vision_config=SimpleNamespace(spatial_merge_size=1),
            ),
        )
        collator.mrope_index.get_rope_index.return_value = (
            torch.zeros((3, 1, 4), dtype=torch.int64),
            None,
        )

        item = _make_instance(None)
        item["audio_feature_lengths"] = torch.tensor([8], dtype=torch.int64)
        item["video_second_per_grid"] = None
        item["input_features"] = torch.zeros((128, 8), dtype=torch.bfloat16)

        with mock.patch.object(
            module,
            "reorganize_inputs",
            return_value=(None, None, torch.zeros((1, 4), dtype=torch.bool)),
        ):
            collator([item], align=4, pack_bin_size=4)

        _args, kwargs = collator.mrope_index.get_rope_index.call_args
        self.assertEqual(tuple(kwargs["attention_mask"].shape), (1, 4))
        self.assertTrue(torch.all(kwargs["attention_mask"] == 1))
        self.assertEqual(int(kwargs["audio_seqlens"].item()), 8)
        self.assertIsNone(kwargs["second_per_grids"])

    def test_flatten_omni_audio_features_drops_masked_frames(self) -> None:
        # flatten 丢掉 pad 帧，lengths 按 clip 计。
        module = self.module
        feat = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.long)
        packed, lengths = module.flatten_omni_audio_features(feat, mask)
        self.assertEqual(tuple(packed.shape), (3, 5))
        self.assertEqual(lengths.tolist(), [3, 2])
        torch.testing.assert_close(packed[:, :3], feat[0, :, :3])
        torch.testing.assert_close(packed[:, 3:], feat[1, :, :2])

    def test_packed_omni_cats_variable_length_audio(self) -> None:
        # WeLM 布局：pack 后是 (mel, T_total) + audio_feature_lengths，不是 per-doc list。
        module = self.module
        collator = module.TrainerV4DataCollatorForQwenVlPacked.__new__(
            module.TrainerV4DataCollatorForQwenVlPacked
        )
        collator.pad_token_id = 0
        collator.vocab_size = 16
        collator.pad_with_random_token = False
        collator.forbidden_token_ids = []
        collator.model_arch = "qwen3_omni_moe"
        collator.mrope_index = mock.Mock(
            hf_class=None,
            config=SimpleNamespace(
                image_token_id=1,
                video_token_id=2,
                vision_config=SimpleNamespace(spatial_merge_size=1),
            ),
        )
        collator.mrope_index.get_rope_index.return_value = (
            torch.zeros((3, 1, 4), dtype=torch.int64),
            None,
        )

        item_a = _make_instance(None)
        item_a["audio_feature_lengths"] = torch.tensor([8], dtype=torch.int64)
        item_a["video_second_per_grid"] = None
        item_a["input_features"] = torch.arange(4 * 8, dtype=torch.bfloat16).reshape(4, 8)
        item_b = _make_instance(None)
        item_b["audio_feature_lengths"] = torch.tensor([16], dtype=torch.int64)
        item_b["video_second_per_grid"] = None
        item_b["input_features"] = torch.arange(4 * 16, dtype=torch.bfloat16).reshape(4, 16) + 100

        with mock.patch.object(
            module,
            "reorganize_inputs",
            return_value=(None, None, torch.zeros((1, 8), dtype=torch.bool)),
        ):
            result = collator([item_a, item_b], align=4, pack_bin_size=8)

        self.assertIsInstance(result["input_features"], torch.Tensor)
        self.assertEqual(tuple(result["input_features"].shape), (4, 24))
        self.assertEqual(result["audio_feature_lengths"].tolist(), [8, 16])
        self.assertNotIn("feature_attention_mask", result)
        torch.testing.assert_close(result["input_features"][:, :8], item_a["input_features"])
        torch.testing.assert_close(result["input_features"][:, 8:], item_b["input_features"])

    def test_packed_omni_skips_text_only_docs_when_cat_audio(self) -> None:
        # pack 里一个有音频、一个纯文本，只 cat 有音频的 doc。
        module = self.module
        collator = module.TrainerV4DataCollatorForQwenVlPacked.__new__(
            module.TrainerV4DataCollatorForQwenVlPacked
        )
        collator.pad_token_id = 0
        collator.vocab_size = 16
        collator.pad_with_random_token = False
        collator.forbidden_token_ids = []
        collator.model_arch = "qwen3_omni_moe"
        collator.mrope_index = mock.Mock(
            hf_class=None,
            config=SimpleNamespace(
                image_token_id=1,
                video_token_id=2,
                vision_config=SimpleNamespace(spatial_merge_size=1),
            ),
        )
        collator.mrope_index.get_rope_index.return_value = (
            torch.zeros((3, 1, 4), dtype=torch.int64),
            None,
        )

        item_audio = _make_instance(None)
        item_audio["audio_feature_lengths"] = torch.tensor([8], dtype=torch.int64)
        item_audio["video_second_per_grid"] = None
        item_audio["input_features"] = torch.arange(4 * 8, dtype=torch.bfloat16).reshape(4, 8)
        item_text = _make_instance(None)
        item_text["audio_feature_lengths"] = None
        item_text["video_second_per_grid"] = None
        item_text["input_features"] = None

        with mock.patch.object(
            module,
            "reorganize_inputs",
            return_value=(None, None, torch.zeros((1, 8), dtype=torch.bool)),
        ):
            result = collator([item_audio, item_text], align=4, pack_bin_size=8)

        self.assertEqual(tuple(result["input_features"].shape), (4, 8))
        self.assertEqual(result["audio_feature_lengths"].tolist(), [8])
        self.assertNotIn("feature_attention_mask", result)

    def test_text_only_grpo_sample_keeps_vision_values_none(self) -> None:
        module = self.module
        collator = module.TrainerV4DataCollatorForQwenVlGRPO.__new__(
            module.TrainerV4DataCollatorForQwenVlGRPO
        )
        collator.dp_rank = 0
        collator.worker_id = 0
        collator.uniq_id = 0
        data = {
            "position_ids": [torch.zeros((3, 1, 4), dtype=torch.int64)],
            "image_input_mask": [torch.zeros((1, 4), dtype=torch.bool)],
            "vision_data": [None],
            "vision_grid_thw": [None],
            "json_data": [{}],
            "tokens": [torch.tensor([5, 6, 0, 0], dtype=torch.int64)],
            "prompt_lengths": [torch.tensor(2, dtype=torch.int64)],
            "imgs_np_array": [None],
        }

        with mock.patch.object(
            module.TrainerV4DataCollatorForQwenVl,
            "__call__",
            return_value=data,
        ):
            result = collator([{}])

        self.assertIsNone(result["vision_data"])
        self.assertIsNone(result["vision_grid_thw"])
