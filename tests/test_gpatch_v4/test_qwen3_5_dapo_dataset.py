import importlib
import sys
import unittest
from types import ModuleType
from unittest import mock

qwenvl_dataset_map = ModuleType("megatron_datasets.qwenvl_dataset_map")
qwenvl_dataset_map.QwenVlDatasetMap = object
qwenvl_dataset_map.TrainerV4DataCollatorForQwenVlGRPO = object
qwenvl_dataset_map.UserQwen2VLImageProcessorFast = object
qwenvl_dataset_map.UserQwen3VLVideoProcessor = object

torch = ModuleType("torch")
torch.utils = ModuleType("torch.utils")
torch.utils.data = ModuleType("torch.utils.data")
torch.utils.data.DataLoader = object
torch.utils.data.Dataset = object
torch.utils.data.distributed = ModuleType("torch.utils.data.distributed")
torch.utils.data.distributed.DistributedSampler = object

transformers = ModuleType("transformers")
transformers.AutoConfig = object
transformers.models = ModuleType("transformers.models")
transformers.models.auto = ModuleType("transformers.models.auto")
transformers.models.auto.processing_auto = ModuleType("transformers.models.auto.processing_auto")
transformers.models.auto.processing_auto.AutoProcessor = object

gpatch_config = ModuleType("gpatch_v4.configs.config")
gpatch_config.RlConfig = object

stub_modules = {
    "datasets": ModuleType("datasets"),
    "torch": torch,
    "torch.utils": torch.utils,
    "torch.utils.data": torch.utils.data,
    "torch.utils.data.distributed": torch.utils.data.distributed,
    "transformers": transformers,
    "transformers.models": transformers.models,
    "transformers.models.auto": transformers.models.auto,
    "transformers.models.auto.processing_auto": transformers.models.auto.processing_auto,
    "gpatch_v4.configs.config": gpatch_config,
    "megatron_datasets.qwenvl_dataset_map": qwenvl_dataset_map,
}
stub_modules["datasets"].load_dataset = object
sys.modules.setdefault("gpatch_v4", ModuleType("gpatch_v4"))
sys.modules.setdefault("gpatch_v4.configs", ModuleType("gpatch_v4.configs"))
with mock.patch.dict("sys.modules", stub_modules):
    dapo_dataset = importlib.import_module("tasks.multimodal_v4.grpo.qwen3_5_dapo_dataset")
convert_dapo_example = dapo_dataset.convert_dapo_example


class Qwen35DapoDatasetTest(unittest.TestCase):
    def test_converts_dapo_prompt_without_changing_fraction_label(self):
        sample = {
            "prompt": [{
                "role": "user",
                "content": "Solve the problem."
            }],
            "label": r"\frac{1}{2}",
        }

        converted = convert_dapo_example(sample, "Keep the answer concise.")

        self.assertEqual(converted["label"], r"\frac{1}{2}")
        self.assertEqual(
            converted["conversations"],
            [
                {
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": "Keep the answer concise."
                    }],
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": "Solve the problem."
                    }],
                },
            ],
        )
        self.assertEqual(converted["images"], [])
        self.assertEqual(converted["__images_feat__"], [])

    def test_rejects_unknown_schema(self):
        with self.assertRaisesRegex(ValueError, "Expected 'prompt' and 'label'"):
            convert_dapo_example({"question": "missing label"}, None)

    def test_rejects_prompt_ending_with_assistant(self):
        sample = {
            "prompt": [{
                "role": "assistant",
                "content": "Already answered."
            }],
            "label": "1",
        }

        with self.assertRaisesRegex(ValueError, "must not end with an assistant"):
            convert_dapo_example(sample, None)


if __name__ == "__main__":
    unittest.main()
