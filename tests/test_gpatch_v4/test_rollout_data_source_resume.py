import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset

from gpatch_v4.orches.data_source import (
    DataSourceBase,
    RolloutDataSource,
    get_data_source,
)
from gpatch_v4.utils.resumable_distributed_sampler import ResumableDistributedSampler
from tasks.math_rl_v4.resumable_rollout_data_source import (
    ResumableRolloutDataSource,
)


class _RangeDataset(Dataset):
    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index


class _RecordingFactory:
    def __init__(self, dataset_size: int = 24):
        self.dataset_size = dataset_size
        self.meta_infos = []

    def __call__(
        self,
        config=None,
        tokenizer=None,
        dp_rank=0,
        dp_size=1,
        meta_info=None,
    ):
        self.meta_infos.append(meta_info)
        dataset = _RangeDataset(self.dataset_size)
        sampler = ResumableDistributedSampler(
            dataset,
            rank=dp_rank,
            num_replicas=dp_size,
            shuffle=False,
            drop_last=True,
        )
        if meta_info is not None:
            rollout_gas = config.training.rollout_gbs // (
                dp_size * config.training.rollout_mbs
            )
            sampler.set_start_index(
                meta_info["resume_step"] * rollout_gas,
                config.training.rollout_mbs,
            )
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=config.training.rollout_mbs,
            drop_last=True,
        )
        return {
            "train_dataset": dataset,
            "train_sampler": sampler,
            "train_dataloader": dataloader,
        }


class RolloutDataSourceResumeTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.factory = _RecordingFactory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _config(self, load_path=None, auto_load=False):
        return SimpleNamespace(
            data=SimpleNamespace(py_path="unused", fn_name="unused"),
            training=SimpleNamespace(
                rollout_mbs=2,
                rollout_gbs=4,
                auto_load_from_save_ckpt=auto_load,
                rollout_ordered_collection=True,
                rollout_max_staleness=1,
            ),
            checkpoint=SimpleNamespace(
                load_ckpt_path=load_path,
                save_ckpt_path=self.tmpdir.name,
            ),
        )

    def _new_source(self, config=None, factory=None):
        with patch(
            "tasks.math_rl_v4.resumable_rollout_data_source.import_fn_from_path",
            return_value=factory or self.factory,
        ):
            return ResumableRolloutDataSource(
                config or self._config(),
                tokenizer=object(),
            )

    def _write_latest_marker(self, step: int):
        marker = os.path.join(self.tmpdir.name, "latest_checkpointed_iteration.txt")
        with open(marker, "w") as f:
            f.write(str(step))

    def _save_checkpoint(self, step: int = 1):
        source = self._new_source()
        source.maybe_set_epoch(0)
        source.get_batch(6)
        source.save(step)
        self._write_latest_marker(step)
        return source

    def test_fresh_source_tracks_consumed_microbatches(self):
        source = self._new_source()

        self.assertEqual(source.dataloader_length, 12)
        self.assertEqual(source.total_dl_consumed, 0)
        source.maybe_set_epoch(0)
        source.get_batch(3)

        self.assertEqual(source.total_dl_consumed, 3)
        self.assertEqual(self.factory.meta_infos, [None])

    def test_save_uses_trained_boundary_instead_of_prefetch_cursor(self):
        source = self._new_source()
        source.maybe_set_epoch(0)
        source.get_batch(6)

        source.save(step=1)

        path = os.path.join(
            self.tmpdir.name,
            "iter_0000001",
            "controller_data_source.pt",
        )
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(
            state,
            {
                "total_dl_consumed": 2,
                "current_epoch": 0,
                "rollout_mbs": 2,
                "dataloader_length": 12,
            },
        )
        self.assertEqual(source.total_dl_consumed, 6)

    def test_init_loads_sidecar_and_passes_resume_step_to_factory(self):
        self._save_checkpoint(step=1)
        resumed_factory = _RecordingFactory()

        source = self._new_source(
            config=self._config(load_path=self.tmpdir.name),
            factory=resumed_factory,
        )

        self.assertEqual(resumed_factory.meta_infos, [{"resume_step": 1}])
        self.assertEqual(source.dataloader_length, 12)
        self.assertEqual(len(source.train_dataloader), 10)
        source.maybe_set_epoch(0)
        batch = source.get_batch(1)[0]
        self.assertEqual(batch.tolist(), [4, 5])
        self.assertEqual(source.total_dl_consumed, 3)

    def test_init_auto_loads_from_save_path(self):
        self._save_checkpoint(step=1)
        resumed_factory = _RecordingFactory()

        self._new_source(
            config=self._config(auto_load=True),
            factory=resumed_factory,
        )

        self.assertEqual(resumed_factory.meta_infos, [{"resume_step": 1}])

    def test_next_epoch_clears_resume_offset(self):
        self._save_checkpoint(step=1)
        source = self._new_source(config=self._config(load_path=self.tmpdir.name))

        self.assertEqual(source.train_sampler.start_index, 4)
        source.maybe_set_epoch(1)

        self.assertEqual(source.train_sampler.start_index, 0)
        self.assertEqual(source.get_batch(1)[0].tolist(), [0, 1])

    def test_resume_requires_sidecar(self):
        self._write_latest_marker(step=1)

        with self.assertRaisesRegex(AssertionError, "missing.*data-source checkpoint"):
            self._new_source(config=self._config(load_path=self.tmpdir.name))

    def test_resume_rejects_rollout_mbs_mismatch(self):
        self._save_checkpoint(step=1)
        path = os.path.join(
            self.tmpdir.name,
            "iter_0000001",
            "controller_data_source.pt",
        )
        state = torch.load(path, map_location="cpu", weights_only=False)
        state["rollout_mbs"] = 4
        torch.save(state, path)

        with self.assertRaisesRegex(AssertionError, "rollout_mbs mismatch"):
            self._new_source(config=self._config(load_path=self.tmpdir.name))

    def test_resume_rejects_unexpected_sidecar_fields(self):
        self._save_checkpoint(step=1)
        path = os.path.join(
            self.tmpdir.name,
            "iter_0000001",
            "controller_data_source.pt",
        )
        state = torch.load(path, map_location="cpu", weights_only=False)
        state["unexpected"] = 1
        torch.save(state, path)

        with self.assertRaisesRegex(AssertionError, "unexpected.*checkpoint fields"):
            self._new_source(config=self._config(load_path=self.tmpdir.name))

    def test_dataloader_remainder_is_allowed(self):
        factory = _RecordingFactory(dataset_size=10)

        source = self._new_source(factory=factory)

        self.assertEqual(source.dataloader_length, 5)

    def test_save_does_not_enforce_ordered_collection(self):
        config = self._config()
        config.training.rollout_ordered_collection = False
        source = self._new_source(config=config)
        source.maybe_set_epoch(0)
        source.get_batch(2)

        source.save(step=1)

        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    self.tmpdir.name,
                    "iter_0000001",
                    "controller_data_source.pt",
                )
            )
        )

    def test_load_allows_unordered_collection(self):
        self._save_checkpoint(step=1)
        config = self._config(load_path=self.tmpdir.name)
        config.training.rollout_ordered_collection = False

        source = self._new_source(config=config)

        self.assertEqual(source.resume_step, 1)


class RolloutDataSourceInjectionTest(unittest.TestCase):
    def test_default_source_is_builtin_rollout_data_source(self):
        config = SimpleNamespace(
            data=SimpleNamespace(
                custom_rollout_data_source_path=None,
                custom_rollout_data_source_name=None,
            )
        )
        tokenizer = object()
        expected_source = object()

        with patch(
            "gpatch_v4.orches.data_source.RolloutDataSource",
            return_value=expected_source,
        ) as data_source_cls:
            source = get_data_source(config, tokenizer)

        self.assertIs(source, expected_source)
        data_source_cls.assert_called_once_with(config, tokenizer)

    def test_custom_source_is_loaded_from_path_and_name(self):
        class CustomDataSource(DataSourceBase):
            def __init__(self, config, tokenizer):
                self.config = config
                self.tokenizer = tokenizer

            def get_batch(self, num_microbatches):
                raise NotImplementedError

            def add_partial_samples(self, samples):
                raise NotImplementedError

            def save(self, step):
                raise NotImplementedError

            def load(self, step):
                raise NotImplementedError

        config = SimpleNamespace(
            data=SimpleNamespace(
                custom_rollout_data_source_path="custom_data_source.py",
                custom_rollout_data_source_name="CustomDataSource",
            )
        )
        tokenizer = object()
        with patch(
            "gpatch_v4.orches.data_source.import_fn_from_path",
            return_value=CustomDataSource,
        ) as import_mock:
            source = get_data_source(config, tokenizer)

        self.assertIsInstance(source, CustomDataSource)
        self.assertIs(source.config, config)
        self.assertIs(source.tokenizer, tokenizer)
        import_mock.assert_called_once_with(
            "custom_data_source.py",
            "CustomDataSource",
        )

    def test_custom_class_must_implement_data_source_interface(self):
        config = SimpleNamespace(
            data=SimpleNamespace(
                custom_rollout_data_source_path="custom_data_source.py",
                custom_rollout_data_source_name="InvalidDataSource",
            )
        )
        with patch(
            "gpatch_v4.orches.data_source.import_fn_from_path",
            return_value=object,
        ):
            with self.assertRaisesRegex(TypeError, "subclass of DataSourceBase"):
                get_data_source(config, object())


if __name__ == "__main__":
    unittest.main()
