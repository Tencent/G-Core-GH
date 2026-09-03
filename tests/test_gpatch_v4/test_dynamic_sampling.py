import asyncio
import os
import shutil
import unittest
from collections import Counter, deque
from contextlib import contextmanager
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.rollout_generator.ds_generator import (
    DynamicSamplingDataSource,
    DynamicSamplingRolloutGenerator,
    _concat_values,
)
from gpatch_v4_test_helper import (
    kill_all_actors_and_shutdown_ray,
    load_config,
    requires_sglang,
)

DS_METRIC_KEYS = [
    "dynamic_sampling/ema_expansion_ratio",
    "dynamic_sampling/num_waves",
    "dynamic_sampling/issued_groups",
    "dynamic_sampling/selected_groups",
    "dynamic_sampling/num_valid_groups",
    "dynamic_sampling/num_invalid_groups",
    "dynamic_sampling/padded_invalid_groups",
    "dynamic_sampling/buffer_size",
]


def _make_filter_generator(advantage_type="grpo", sampling_repeat=2, dynamic_filter=None):
    generator = object.__new__(DynamicSamplingRolloutGenerator)
    generator.config = SimpleNamespace(ppo=SimpleNamespace(advantage_type=advantage_type))
    generator.sampling_repeat = sampling_repeat
    generator.dynamic_filter = dynamic_filter
    return generator


def _reward_group(*values):
    return {"rewards": [torch.tensor(value) for value in values]}


def test_filter_group_equal_rewards():
    generator = _make_filter_generator()
    keep, reason = generator._filter_group(_reward_group(1.0, 1.0))
    assert keep is False
    assert reason == "equal_rewards"


def test_filter_group_valid():
    generator = _make_filter_generator()
    keep, reason = generator._filter_group(_reward_group(1.0, 0.0))
    assert keep is True
    assert reason == "valid"


def test_filter_group_no_filter():
    generator = _make_filter_generator(advantage_type="ppo")
    keep, reason = generator._filter_group(_reward_group(1.0, 1.0))
    assert keep is True
    assert reason == "no_filter"


def test_filter_group_custom_filter():
    def drop_all(config, group):
        return False, "custom_drop"

    generator = _make_filter_generator(dynamic_filter=drop_all)
    keep, reason = generator._filter_group(_reward_group(1.0, 0.0))
    assert keep is False
    assert reason == "custom_drop"


def test_datasource_wraps_epoch_and_sets_limit():
    epochs_seen = []

    def reset_iter(epoch, skip_batches=0):
        epochs_seen.append(epoch)
        data = [{"epoch": epoch, "idx": 0}, {"epoch": epoch, "idx": 1}]
        return iter(data[skip_batches:])

    source = DynamicSamplingDataSource(
        reset_iter,
        range(2),
        current_epoch=0,
        max_epochs=2,
        batch_size=1,
    )
    batches = [source.next_batch() for _ in range(4)]
    assert [batch["epoch"] for batch in batches] == [0, 0, 1, 1]
    assert epochs_seen == [0, 1]
    assert source.current_epoch == 1
    assert source.reached_epoch_limit is True


def test_split_concat_roundtrip():
    rollout_batch = {
        "tokens": [[0], [1], [2], [3]],
        "unique_id": ["a", "a", "b", "b"],
        "nested": {"x": [10, 11, 12, 13]},
        "scalar": 7,
    }
    groups = DynamicSamplingRolloutGenerator._split_rollout_batch(
        rollout_batch, num_prompts=2, repeat_n=2
    )
    assert len(groups) == 2
    assert groups[0]["tokens"] == [[0], [1]]
    assert groups[1]["unique_id"] == ["b", "b"]
    assert groups[0]["nested"]["x"] == [10, 11]
    assert groups[1]["scalar"] == 7

    merged = {
        key: _concat_values([group[key] for group in groups])
        for key in rollout_batch
    }
    assert merged == rollout_batch


class _FakeTimer:
    def __call__(self, *args, **kwargs):
        return self

    def start(self, *args, **kwargs):
        return None

    def stop(self, *args, **kwargs):
        return None


class _FakeSamplerClient:
    async def mark_ppo_step_begin(self, *args, **kwargs):
        return None

    async def infer_engine_flush_cache(self, *args, **kwargs):
        return None

    async def mark_ppo_step_end(self, *args, **kwargs):
        return None


class _FakeRolloutAttr:
    def __init__(self):
        self.cache = {}

    def cached_rollout_attrs(self):
        return self.cache

    def remove_rollout_attr_before_sampling(self, rollout_batch):
        for unique_id in rollout_batch["unique_id"]:
            assert unique_id not in self.cache
            self.cache[unique_id] = {}
        return rollout_batch


class _FakePromptSource:
    def __init__(self):
        self.n_called = 0

    def next_batch(self):
        uid = f"p{self.n_called}"
        self.n_called += 1
        return {
            "unique_id": [uid],
            "tokens": [[self.n_called]],
            "prompt_len": [torch.tensor(1)],
        }


class _FakePromptIter:
    def __init__(self, source):
        self.source = source

    def __iter__(self):
        return self

    def __next__(self):
        return self.source.next_batch()


def _wrap_ds_source(source=None, batch_size=1, batches_per_epoch=10**6, max_epochs=100):
    source = source or _FakePromptSource()

    def reset_iter(epoch, skip_batches=0):
        return _FakePromptIter(source)

    ds = DynamicSamplingDataSource(
        reset_iter,
        range(batches_per_epoch),
        current_epoch=0,
        max_epochs=max_epochs,
        batch_size=batch_size,
    )
    return source, ds


def _cpu_update_ema(self, num_valid_groups, num_invalid_groups):
    total_groups = num_valid_groups + num_invalid_groups
    assert total_groups > 0
    if num_valid_groups == 0:
        current_ratio = self.dynamic_config.max_expansion_ratio
    else:
        current_ratio = min(
            total_groups / num_valid_groups,
            self.dynamic_config.max_expansion_ratio,
        )
    self.ema_expansion_ratio = (
        self.dynamic_config.ema_decay * self.ema_expansion_ratio +
        (1.0 - self.dynamic_config.ema_decay) * current_ratio
    )


def _cpu_collect_step_metrics(
    self,
    num_waves,
    num_issued_groups,
    num_selected_groups,
    num_valid_groups,
    num_invalid_groups,
    num_padded_invalid_groups,
):
    merged_reasons = Counter(self.filter_reasons)
    self.filter_reasons.clear()
    metrics = {
        "dynamic_sampling/ema_expansion_ratio": self.ema_expansion_ratio,
        "dynamic_sampling/num_waves": float(num_waves),
        "dynamic_sampling/issued_groups": float(num_issued_groups),
        "dynamic_sampling/selected_groups": float(num_selected_groups),
        "dynamic_sampling/num_valid_groups": float(num_valid_groups),
        "dynamic_sampling/num_invalid_groups": float(num_invalid_groups),
        "dynamic_sampling/padded_invalid_groups": float(num_padded_invalid_groups),
        "dynamic_sampling/buffer_size": float(len(self.data_source.prompt_buffer)),
    }
    for reason, count in merged_reasons.items():
        metrics[f"dynamic_sampling/filter_reason/{reason}"] = float(count)
    return metrics


def _expand_with_rewards(orig_batches, repeat_n, reward_fn):
    rbs = []
    for orig_batch in orig_batches:
        rollout_batch = {}
        for key, value in orig_batch.items():
            if isinstance(value, list):
                expanded = []
                for item in value:
                    expanded.extend([item] * repeat_n)
                rollout_batch[key] = expanded
            else:
                rollout_batch[key] = value
        rewards = []
        for unique_id in orig_batch["unique_id"]:
            group_rewards = reward_fn(unique_id)
            assert len(group_rewards) == repeat_n
            rewards.extend(group_rewards)
        rollout_batch["rewards"] = [torch.tensor(reward) for reward in rewards]
        rbs.append(rollout_batch)
    return rbs


def _valid_rewards(_uid):
    return [1.0, 0.0]


def _invalid_rewards(_uid):
    return [1.0, 1.0]


def _make_ds_generator(
    reward_fn=_valid_rewards,
    oversampling_ratio=1.0,
    init_expansion_ratio=1.0,
    ema_decay=0.9,
    max_expansion_ratio=10.0,
    max_refill_times=3,
    dynamic_filter=None,
):
    generator = object.__new__(DynamicSamplingRolloutGenerator)
    generator.config = SimpleNamespace(
        ppo=SimpleNamespace(advantage_type="grpo"),
        placement_type="colocate",
        training=SimpleNamespace(
            rollout_mbs=1,
            sampling_repeat_n=2,
            offload_process_group=False,
            use_external_reward=False,
            use_gen_rm_reward=False,
            use_bt_rm_reward=False,
            dynamic_sampling=SimpleNamespace(
                oversampling_ratio=oversampling_ratio,
                ema_decay=ema_decay,
                init_expansion_ratio=init_expansion_ratio,
                max_expansion_ratio=max_expansion_ratio,
                max_refill_times=max_refill_times,
                filter_py_path=None,
                filter_fn_name=None,
            ),
        ),
    )
    generator.dynamic_config = generator.config.training.dynamic_sampling
    generator.ema_expansion_ratio = init_expansion_ratio
    generator.prompt_issue_idx = 0
    generator.request_idx = 0
    generator.filter_reasons = Counter()
    generator._step_metrics = {}
    generator.data_source = None
    generator.dynamic_filter = dynamic_filter
    generator.sampling_repeat = 2
    generator.sample_idx = 0
    generator.run_eval = False
    generator.is_mp_and_cp_head = True
    generator.external_reward = None
    generator.sampler_client = _FakeSamplerClient()
    generator.apply_sampling_rollout_attr = _FakeRolloutAttr()
    generator.assign_unique_id_to_batches = lambda batch, dp_rank, rbi: batch
    generator.remove_rollout_attr_before_sampling = (
        generator.apply_sampling_rollout_attr.remove_rollout_attr_before_sampling
    )
    generator._post_process_rm_rollout_batch = lambda rbs: rbs
    generator._update_ema = MethodType(_cpu_update_ema, generator)
    generator._collect_step_metrics = MethodType(_cpu_collect_step_metrics, generator)

    async def sampler_gen_out(
        orig_batches, sampler_idx, curr_ppo_step, sample_idx, repeat_n, on_ready=None
    ):
        return _expand_with_rewards(orig_batches, repeat_n, reward_fn)

    generator.sampler_gen_out = sampler_gen_out
    return generator


@contextmanager
def _ds_call_patches():
    with patch("gpatch_v4.rollout_generator.ds_generator.cpu_barrier"), \
         patch("gpatch_v4.rollout_generator.ds_generator.logging_memory_usage_details"), \
         patch("gpatch_v4.rollout_generator.ds_generator.mpu") as mpu, \
         patch("gpatch_v4.rollout_generator.ds_generator.TimerSingleton") as timer_cls, \
         patch.object(torch.distributed, "is_initialized", return_value=False):
        mpu.get_data_parallel_rank.return_value = 0
        timer_cls.get_timer.return_value = _FakeTimer()
        yield


def _run_ds_step(generator, data_iter, num_microbatches=2):
    generator.data_source = data_iter

    async def _run():
        with _ds_call_patches():
            return await generator(data_iter, num_microbatches, 0)

    return asyncio.run(_run())


def test_datasource_wraps_past_max_after_limit_is_set():
    epochs_seen = []

    def reset_iter(epoch, skip_batches=0):
        epochs_seen.append(epoch)
        data = [{"epoch": epoch, "idx": 0}, {"epoch": epoch, "idx": 1}]
        return iter(data[skip_batches:])

    source = DynamicSamplingDataSource(
        reset_iter,
        range(2),
        current_epoch=0,
        max_epochs=2,
        batch_size=1,
    )
    batches = [source.next_batch() for _ in range(4)]
    assert [batch["epoch"] for batch in batches] == [0, 0, 1, 1]
    assert source.reached_epoch_limit is True
    extra = source.next_batch()
    assert extra["epoch"] == 2
    assert source.current_epoch == 2
    assert epochs_seen == [0, 1, 2]


def test_ds_one_wave_all_valid():
    generator = _make_ds_generator()
    source, data = _wrap_ds_source()
    rbs = _run_ds_step(generator, data)
    metrics = generator.pop_step_metrics()
    assert len(rbs) == 2
    assert source.n_called == 2
    assert metrics["dynamic_sampling/num_waves"] == 1
    assert metrics["dynamic_sampling/issued_groups"] == 2
    assert metrics["dynamic_sampling/selected_groups"] == 2
    assert metrics["dynamic_sampling/num_valid_groups"] == 2
    assert metrics["dynamic_sampling/num_invalid_groups"] == 0
    assert metrics["dynamic_sampling/padded_invalid_groups"] == 0
    assert metrics["dynamic_sampling/buffer_size"] == 0
    assert metrics["dynamic_sampling/ema_expansion_ratio"] == 1.0
    assert "sample_mask" not in rbs[0]
    assert set(generator.apply_sampling_rollout_attr.cache) == {"p0", "p1"}


def test_ds_refill_then_ema_from_mixed_validity():
    def reward_fn(uid):
        if uid == "p0":
            return [1.0, 1.0]
        return [1.0, 0.0]

    generator = _make_ds_generator(reward_fn=reward_fn)
    source, data = _wrap_ds_source()
    rbs = _run_ds_step(generator, data)
    metrics = generator.pop_step_metrics()
    assert [rb["unique_id"][0] for rb in rbs] == ["p1", "p2"]
    assert source.n_called == 3
    assert metrics["dynamic_sampling/num_waves"] == 2
    assert metrics["dynamic_sampling/issued_groups"] == 3
    assert metrics["dynamic_sampling/num_valid_groups"] == 2
    assert metrics["dynamic_sampling/num_invalid_groups"] == 1
    assert metrics["dynamic_sampling/padded_invalid_groups"] == 0
    assert metrics["dynamic_sampling/filter_reason/equal_rewards"] == 1
    assert metrics["dynamic_sampling/filter_reason/valid"] == 2
    assert abs(metrics["dynamic_sampling/ema_expansion_ratio"] - 1.05) < 1e-6
    assert "p0" not in generator.apply_sampling_rollout_attr.cache
    assert set(generator.apply_sampling_rollout_attr.cache) == {"p1", "p2"}


def test_ds_last_wave_pads_invalid_groups():
    generator = _make_ds_generator(
        reward_fn=_invalid_rewards,
        max_refill_times=0,
    )
    _, data = _wrap_ds_source()
    rbs = _run_ds_step(generator, data)
    metrics = generator.pop_step_metrics()
    assert len(rbs) == 2
    assert metrics["dynamic_sampling/num_waves"] == 1
    assert metrics["dynamic_sampling/num_valid_groups"] == 0
    assert metrics["dynamic_sampling/num_invalid_groups"] == 2
    assert metrics["dynamic_sampling/padded_invalid_groups"] == 2
    assert abs(metrics["dynamic_sampling/ema_expansion_ratio"] - 1.9) < 1e-6
    for rb in rbs:
        assert [mask.item() for mask in rb["sample_mask"]] == [False, False]
    assert set(generator.apply_sampling_rollout_attr.cache) == {"p0", "p1"}


def test_ds_pads_after_refill_budget_not_only_when_max_refill_zero():
    def reward_fn(uid):
        if uid == "p2":
            return [1.0, 0.0]
        return [1.0, 1.0]

    generator = _make_ds_generator(reward_fn=reward_fn, max_refill_times=1)
    _, data = _wrap_ds_source()
    rbs = _run_ds_step(generator, data)
    metrics = generator.pop_step_metrics()
    assert metrics["dynamic_sampling/num_waves"] == 2
    assert metrics["dynamic_sampling/num_valid_groups"] == 1
    assert metrics["dynamic_sampling/padded_invalid_groups"] == 1
    assert metrics["dynamic_sampling/num_invalid_groups"] == 3
    masks = []
    for rb in rbs:
        masks.append(rb["sample_mask"][0].item())
    assert True in masks
    assert False in masks
    assert "p1" not in generator.apply_sampling_rollout_attr.cache
    assert "p3" not in generator.apply_sampling_rollout_attr.cache
    assert set(generator.apply_sampling_rollout_attr.cache) == {"p0", "p2"}


def test_ds_custom_filter_runs_inside_wave():
    def drop_zero_first_reward(config, group):
        if group["rewards"][0].item() == 0.0:
            return False, "custom_drop"
        return True, "valid"

    def reward_fn(uid):
        if uid == "p0":
            return [0.0, 1.0]
        return [1.0, 0.0]

    generator = _make_ds_generator(
        reward_fn=reward_fn,
        dynamic_filter=drop_zero_first_reward,
        max_refill_times=1,
    )
    rbs = _run_ds_step(generator, _wrap_ds_source()[1])
    metrics = generator.pop_step_metrics()
    assert [rb["unique_id"][0] for rb in rbs] == ["p1", "p2"]
    assert metrics["dynamic_sampling/filter_reason/custom_drop"] == 1
    assert metrics["dynamic_sampling/filter_reason/valid"] == 2
    assert "dynamic_sampling/filter_reason/equal_rewards" not in metrics


def test_ds_oversample_leftover_stays_in_buffer_until_full_wave():
    generator = _make_ds_generator(oversampling_ratio=2.0)
    source, data = _wrap_ds_source()
    _run_ds_step(generator, data)
    metrics = generator.pop_step_metrics()
    assert metrics["dynamic_sampling/issued_groups"] == 4
    assert metrics["dynamic_sampling/selected_groups"] == 2
    assert metrics["dynamic_sampling/num_valid_groups"] == 4
    assert metrics["dynamic_sampling/buffer_size"] == 2
    leftover = list(data.prompt_buffer)
    assert source.n_called == 4

    with _ds_call_patches():
        issue_ids, batches = generator._get_wave_batches(data, gap=2, dp_rank=0)
    assert source.n_called == 8
    assert len(issue_ids) == 4
    assert len(data.prompt_buffer) == 2
    assert leftover == list(data.prompt_buffer)
    assert batches[0]["unique_id"][0] == "p4"


def test_ds_wave_drains_buffer_when_it_covers_the_wave():
    generator = _make_ds_generator()
    source, data = _wrap_ds_source()
    for i in range(2):
        data.put_back({
            "unique_id": [f"buf{i}"],
            "tokens": [[i]],
            "prompt_len": [torch.tensor(1)],
        })
    with _ds_call_patches():
        issue_ids, batches = generator._get_wave_batches(data, gap=2, dp_rank=0)
    assert source.n_called == 0
    assert data.prompt_buffer == deque()
    assert [ids[0] for ids in issue_ids] == [0, 1]
    assert [batch["unique_id"][0] for batch in batches] == ["buf0", "buf1"]


def test_datasource_take_wave_prefers_buffer_when_full():
    source, data = _wrap_ds_source()
    for i in range(2):
        data.put_back({"tokens": [[i]], "idx": [i]})
    batches = data.take_wave(2)
    assert source.n_called == 0
    assert [batch["idx"][0] for batch in batches] == [0, 1]
    assert len(data.prompt_buffer) == 0
    batches = data.take_wave(1)
    assert source.n_called == 1
    assert batches[0]["unique_id"] == ["p0"]


def test_datasource_state_dict_roundtrip_restores_cursor_and_buffer():
    calls = []

    def reset_iter(epoch, skip_batches=0):
        calls.append((epoch, skip_batches))
        data = [{"idx": [i]} for i in range(4)]
        return iter(data[skip_batches:])

    source = DynamicSamplingDataSource(
        reset_iter,
        range(4),
        current_epoch=0,
        max_epochs=3,
        batch_size=1,
    )
    source.next_batch()
    source.next_batch()
    source.put_back({"tokens": [[9]], "idx": [9]})
    state = source.state_dict()
    assert state["current_epoch"] == 0
    assert state["batches_consumed"] == 2
    assert state["reached_epoch_limit"] is False
    assert len(state["prompt_buffer"]) == 1

    restored = DynamicSamplingDataSource(
        reset_iter,
        range(4),
        current_epoch=0,
        max_epochs=3,
        batch_size=1,
    )
    restored.load_state_dict(state)
    assert restored.current_epoch == 0
    assert restored.batches_consumed == 2
    assert restored.reached_epoch_limit is False
    assert list(restored.prompt_buffer) == [{"tokens": [[9]], "idx": [9]}]
    assert calls[-1] == (0, 2)
    assert restored.next_batch()["idx"] == [2]


class DynamicSamplingGpuTest(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    @requires_sglang
    async def test_train_sglang(self):
        from gpatch_v4.trainer import GrpoTrainer

        config = load_config("test_math_rl", RlConfig)
        config.policy.rollout_gen_type = "dynamic_sampling"
        config.training.dynamic_sampling.max_refill_times = 3
        config.training.total_ppo_step = 1

        tmp_dir = "tests/test_gpatch_v4/gsm8k_small"
        os.makedirs(tmp_dir, exist_ok=True)
        src = "hf-hub/DaertML/gsm8k-jsonl/train.jsonl"
        dst = os.path.join(tmp_dir, "train.jsonl")
        with open(src, "r") as fin, open(dst, "w") as fout:
            for i, line in enumerate(fin):
                if i >= 256:
                    break
                fout.write(line)

        shutil.rmtree(config.checkpoint.load_ckpt_path, ignore_errors=True)
        try:
            trainer = GrpoTrainer()
            metrics = await trainer.launch_then_run_with_recovery(config)
        finally:
            shutil.rmtree(config.checkpoint.save_ckpt_path, ignore_errors=True)

        assert metrics is not None
        assert len(metrics) > 0
        for dp_metrics in metrics:
            assert len(dp_metrics) >= 1
            for step_metric in dp_metrics:
                assert "policy/loss" in step_metric
                assert "policy/grad_norm" in step_metric
                assert "policy/ppo_ratio" in step_metric
                loss = step_metric["policy/loss"]
                grad_norm = step_metric["policy/grad_norm"]
                ppo_ratio = step_metric["policy/ppo_ratio"]
                assert -0.01 < loss < 0.01, f"policy/loss out of range: {loss}"
                assert 0 <= grad_norm < 1.0, f"policy/grad_norm out of range: {grad_norm}"
                assert 0.5 < ppo_ratio < 1.01, f"policy/ppo_ratio out of range: {ppo_ratio}"
                for key in DS_METRIC_KEYS:
                    assert key in step_metric, f"missing {key} in {step_metric.keys()}"
