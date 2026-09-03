"""TailBatchingDataSource 的纯 CPU 单测。"""

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from gpatch_v4.orches.data_source import RolloutDataSource, TailBatchingDataSource
from gpatch_v4.utils.resumable_distributed_sampler import ResumableDistributedSampler


class _FakeSampler:
    def __init__(self):
        self.resume_calls = []
        self.epoch = 0
        self.epoch_calls = []
        self.start_index = 0

    def set_start_index(self, start_step: int, batch_size: int):
        self.resume_calls.append((start_step, batch_size))

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.epoch_calls.append(epoch)


class _FakeDataLoader:
    def __iter__(self):
        return iter([])

    def __len__(self) -> int:
        return 64


class _StubTailBatchingDataSource(TailBatchingDataSource):
    def _build_dataset_and_dataloader(self):
        self.train_dataset = list(range(64))
        self.train_dataloader = _FakeDataLoader()
        self.train_sampler = _FakeSampler()


class _StubRolloutDataSource(RolloutDataSource):
    def _build_dataset_and_dataloader(self):
        self.train_dataset = list(range(64))
        self.train_dataloader = _FakeDataLoader()
        self.train_sampler = _FakeSampler()


def _make_source(
    tmp_path,
    load_path=None,
    rollout_mbs: int = 2,
    rollout_over_dispatch_ratio: float = 1.5,
):
    config = SimpleNamespace(
        checkpoint=SimpleNamespace(
            save_ckpt_path=str(tmp_path),
            load_ckpt_path=str(tmp_path if load_path is None else load_path),
        ),
        training=SimpleNamespace(
            rollout_mbs=rollout_mbs,
            rollout_over_dispatch_ratio=rollout_over_dispatch_ratio,
            rollout_reuse_unused_prompts=True,
        ),
    )
    return _StubTailBatchingDataSource(config, tokenizer=None)


def test_get_batch_buffer_hit_and_over_fire_miss(tmp_path):
    ds = _make_source(tmp_path)
    ds.add_partial_samples([{"value": [i]} for i in range(5)])
    # _train_iter 仍是 None，一旦误读 dataloader，父类断言会失败
    batches = ds.get_batch(3)
    assert [b["value"] for b in batches] == [[0], [1], [2]]
    assert ds.buffer_length == 2
    assert ds.total_dl_consumed == 0

    ds = _make_source(tmp_path)
    ds.add_partial_samples([{"value": [0]}])
    ds._train_iter = iter([{"value": [100 + i]} for i in range(6)])
    batches = ds.get_batch(4)
    assert len(batches) == 6
    assert ds.buffer_length == 1
    assert ds.total_dl_consumed == 6


def test_maybe_set_epoch_keeps_buffer_and_iterator(tmp_path):
    ds = _make_source(tmp_path)
    ds.add_partial_samples([{"value": [1]}])
    ds.maybe_set_epoch(0)
    train_iter = ds._train_iter

    ds.maybe_set_epoch(1)

    assert ds.buffer_length == 1
    assert ds._current_epoch == 0
    assert ds._train_iter is train_iter


def test_base_data_source_still_resets_on_trainer_epoch_change():
    ds = _StubRolloutDataSource(SimpleNamespace(), tokenizer=None)
    ds.maybe_set_epoch(0)
    train_iter = ds._train_iter

    ds.maybe_set_epoch(1)

    assert ds._current_epoch == 1
    assert ds._train_iter is not train_iter
    assert ds.train_sampler.epoch_calls == [0, 1]


def test_get_batch_crosses_data_epoch_and_fills_requested_size(tmp_path):
    ds = _make_source(tmp_path)
    ds.train_dataloader = [
        {"value": [0]},
        {"value": [1]},
        {"value": [2]},
    ]
    ds.maybe_set_epoch(0)

    batches = ds.get_batch(2)
    assert [batch["value"] for batch in batches] == [[0], [1], [2]]

    batches = ds.get_batch(2)
    assert [batch["value"] for batch in batches] == [[0], [1], [2]]
    assert ds._current_epoch == 1
    assert ds.train_sampler.epoch_calls == [0, 1]


def test_data_epoch_advance_clears_resume_offset(tmp_path):
    ds = _make_source(tmp_path)
    dataset = list(range(3))
    sampler = ResumableDistributedSampler(
        dataset,
        num_replicas=1,
        rank=0,
        shuffle=False,
        drop_last=True,
    )
    sampler.set_start_index(2, 1)
    ds.train_sampler = sampler
    ds.train_dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)
    ds._current_epoch = 0
    ds._train_iter = iter(ds.train_dataloader)

    batches = ds.get_batch(2)

    assert [batch.item() for batch in batches] == [2, 0, 1]
    assert ds._current_epoch == 1
    assert sampler.epoch == 1
    assert sampler.start_index == 0


def test_save_then_load_restores_buffer_and_sampler_offset(tmp_path):
    ds = _make_source(tmp_path)
    ds._total_dl_consumed = 12
    ds._current_epoch = 1
    ds.add_partial_samples([{"value": torch.tensor([7])}])

    ds.save(5, pending_unconsumed_batches=[{"value": torch.tensor([8, 9])}])

    state_path = tmp_path / "iter_0000005" / "controller_data_source.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    assert state["total_dl_consumed"] == 12
    assert state["current_epoch"] == 1
    assert state["rollout_mbs"] == 2
    assert [batch["value"].tolist() for batch in state["buffer"]] == [[7], [8, 9]]

    restored = _make_source(tmp_path)
    restored.load(5)
    assert restored.total_dl_consumed == 12
    assert restored._current_epoch == 1
    assert restored.train_sampler.resume_calls == [(12, 2)]
    assert [batch["value"].tolist() for batch in restored._buffer] == [[7], [8, 9]]
    assert restored._train_iter is not None


def test_save_replaces_final_file_only_after_torch_save(tmp_path, monkeypatch):
    ds = _make_source(tmp_path)
    state_path = tmp_path / "iter_0000005" / "controller_data_source.pt"
    real_torch_save = torch.save

    def checked_save(state, path):
        assert str(path).endswith(".tmp")
        assert not state_path.exists()
        real_torch_save(state, path)

    monkeypatch.setattr(torch, "save", checked_save)
    ds.save(5)

    assert state_path.exists()
    assert not state_path.with_suffix(".pt.tmp").exists()


def test_load_requires_checkpoint_file(tmp_path):
    with pytest.raises(AssertionError, match="missing tail-batching data-source checkpoint"):
        _make_source(tmp_path).load(5)


def test_load_rejects_rollout_mbs_mismatch(tmp_path):
    _make_source(tmp_path, rollout_mbs=2).save(5)

    with pytest.raises(AssertionError, match="rollout_mbs mismatch"):
        _make_source(tmp_path, rollout_mbs=4).load(5)


def test_load_rejects_unexpected_checkpoint_fields(tmp_path):
    state_dir = tmp_path / "iter_0000005"
    state_dir.mkdir()
    torch.save({"buffer": []}, state_dir / "controller_data_source.pt")

    with pytest.raises(AssertionError, match="unexpected tail-batching checkpoint fields"):
        _make_source(tmp_path).load(5)


def test_load_reads_from_load_ckpt_path_not_save(tmp_path):
    old = tmp_path / "old_run"
    new = tmp_path / "new_run"
    old.mkdir()
    new.mkdir()

    seed = _make_source(old)
    seed._total_dl_consumed = 20
    seed.save(3)

    resumed = _make_source(new, load_path=old)
    resumed.load(3)
    assert resumed.total_dl_consumed == 20
    assert resumed.train_sampler.resume_calls == [(20, 2)]
