# copyright (c) 2024 tencent inc. all rights reserved.
# guanyouhe@tencent.com
"""Savable Energon dataset + dataloader from ``path`` + ``map_func``.

Typical non-packed v4 usage::

    from gdataset.energon_dataset import (
        get_non_packed_energon_dataset_and_dataloader,
    )

    def get_dataset_and_dataloader(
        config, tokenizer=None, dp_rank=0, dp_size=1, meta_info=None
    ):
        map_func = QwenVlDatasetMap(...)  # or WeLMOmniMap(...)
        return get_non_packed_energon_dataset_and_dataloader(
            config,
            map_func=map_func,
            sample_process_func=process_sample,
            tokens_key="input_ids",
            dp_rank=dp_rank,
            dp_size=dp_size,
            meta_info=meta_info,
            collate_func=TrainerV4DataCollatorForQwenVl(...),  # optional
        )

``data.data_pathes[0]`` may be a prepared WebDataset directory (``.nv-meta``)
or a Metadataset yaml. Set ``task.pack_bin_size`` to enable packing.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import torch
import yaml

from megatron.energon import (
    BatchDataset,
    Cooker,
    CrudeSample,
    DefaultTaskEncoder,
    MapDataset,
    Sample,
    SkipSample,
    WorkerConfig,
    basic_sample_keys,
    edataclass,
    get_savable_loader,
    get_train_dataset,
    stateless,
)
from megatron.energon.epathlib import EPath
from megatron.energon.errors import FatalSampleError
from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.flavors.jsonl.ijsonl import IJsonlIndexReader
from megatron.energon.flavors.webdataset.metadata import (
    EnergonDatasetType,
    get_dataset_info,
    get_dataset_type,
)
from megatron.energon.metadataset.loader import traverse_metadataset
from megatron.energon.task_encoder.base import get_failure_tolerance, get_stateless
from megatron.energon.wrappers.buffer import SavableSampleBuffer
from megatron.energon.wrappers.packing_dataset import PackingDataset

T_sample = Any
T_encoded_sample = Any
T_batch_sample = Any


@edataclass
class RawSample(Sample):
    data: dict


@edataclass
class BatchedSample(Sample):
    data: dict


def resolve_energon_data_path(data_path: str) -> str:
    """Return the path Energon ``get_train_dataset`` should open."""
    assert data_path, "data.data_pathes[0] must be a non-empty energon path"
    if os.path.isfile(data_path):
        if data_path.endswith((".yaml", ".yml")):
            return data_path
        if data_path.endswith(".jsonl") and os.path.isfile(f"{data_path}.idx"):
            return data_path
        raise AssertionError(
            "unsupported energon file (want prepared .jsonl or metadataset "
            f".yaml/.yml): {data_path}"
        )

    if os.path.isdir(data_path):
        nv_meta = os.path.join(data_path, ".nv-meta")
        if os.path.isdir(nv_meta):
            return data_path
        for name in ("metadataset.yaml", "energon_meta.yaml"):
            candidate = os.path.join(data_path, name)
            if os.path.isfile(candidate):
                return candidate
        raise AssertionError(
            f"{data_path} is a directory but has neither .nv-meta nor "
            "metadataset.yaml / energon_meta.yaml; prepare the dataset first"
        )
    raise AssertionError(f"energon dataset path not found: {data_path}")


def _sum_dataset_samples(paths: Iterable[str]) -> int:
    total = 0
    for path in paths:
        energon_path = EPath(str(path))
        if get_dataset_type(energon_path) == EnergonDatasetType.JSONL:
            total += IJsonlIndexReader.count_samples(energon_path)
            continue
        info = get_dataset_info(energon_path)
        shard_counts = info.get("shard_counts") or {}
        total += int(sum(shard_counts.values()))
    return total


def _estimate_metadataset_samples(mds_path: str) -> int:
    try:
        refs = traverse_metadataset(EPath(mds_path), split_part="train")
        dataset_paths = [str(ref.path) for ref in refs]
        total = _sum_dataset_samples(dataset_paths)
        if total > 0:
            return total
    except Exception as exc:  # noqa: BLE001
        print(f"warn: traverse_metadataset failed ({exc}); trying yaml parse fallback")

    with open(mds_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    blend = cfg["splits"]["train"]["blend"]
    parent = Path(mds_path).parent
    dataset_paths = [str(parent / entry["path"]) for entry in blend]
    total = _sum_dataset_samples(dataset_paths)
    assert total > 0, f"metadataset has no samples: {mds_path}"
    return total


def estimate_num_samples(data_path: str) -> int:
    """Count samples for epoch sizing (Energon itself is cyclic)."""
    resolved = resolve_energon_data_path(data_path)
    ds_type = get_dataset_type(EPath(resolved))
    if ds_type == EnergonDatasetType.METADATASET:
        return _estimate_metadataset_samples(resolved)
    if ds_type == EnergonDatasetType.JSONL:
        return IJsonlIndexReader.count_samples(EPath(resolved))
    if ds_type == EnergonDatasetType.WEBDATASET:
        try:
            total = _sum_dataset_samples([resolved])
            if total > 0:
                return total
        except Exception as exc:  # noqa: BLE001
            print(f"warn: cannot read energon info for length estimate: {exc}")
        return 10**9
    raise RuntimeError(f"unsupported energon dataset type {ds_type}: {resolved}")


def collate_fn_packed(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate packed microbatches; ``train_mbs=1`` → the packed sample itself."""
    assert examples, "empty packed collate"
    assert len(examples) == 1, (f"packing collate expects train_mbs=1, got {len(examples)}")
    return examples[0]


def _dataloader_save_path(checkpoint_config, iteration, dp_rank):
    """Match ``gpatch_v4...checkpoint.get_dataloader_save_path``."""
    save_ckpt_path = checkpoint_config.save_ckpt_path
    if not save_ckpt_path:
        return None, None
    out_dir = os.path.join(save_ckpt_path, f"dataloader/iter_{iteration:07d}")
    state_path = os.path.join(out_dir, f"dp_rank_{dp_rank:03d}.pt")
    return out_dir, state_path


def normalize_energon_ckpt_state(state: Any) -> Dict[str, Any]:
    """Normalize a torch-loaded Energon ckpt into restore_state() input."""
    if isinstance(state, dict) and "dataloader_state_dict" in state:
        metadata = state.get("gcore_checkpoint_metadata") or {}
        inner = state["dataloader_state_dict"]
        if isinstance(inner, dict) and "loader_state" in inner:
            metadata = inner.get("checkpoint_metadata") or metadata or {}
            return {
                "loader_state": inner["loader_state"],
                "checkpoint_metadata": metadata,
            }
        return {
            "loader_state": inner,
            "checkpoint_metadata": metadata if isinstance(metadata, dict) else {},
        }
    if isinstance(state, dict) and "loader_state" in state:
        metadata = state.get("checkpoint_metadata") or {}
        assert isinstance(metadata,
                          dict), (f"checkpoint_metadata must be a dict, got {type(metadata)}")
        return {
            "loader_state": state["loader_state"],
            "checkpoint_metadata": metadata,
        }
    return {"loader_state": state, "checkpoint_metadata": {}}


def maybe_restore_dataloader(
    dataloader: EnergonDataloader,
    config,
    dp_rank: int,
    dp_size: int,
    meta_info: Optional[Dict[str, Any]],
    *,
    require_pretrain_metadata: bool = False,
    expected_num_samples: Optional[int] = None,
) -> None:
    resume_step = None
    if meta_info is not None and "resume_step" in meta_info:
        resume_step = int(meta_info["resume_step"])
    if resume_step is None or resume_step <= 0:
        print(
            f"[energon] no dataloader restore: "
            f"resume_step={resume_step} dp={dp_rank}/{dp_size}",
            flush=True,
        )
        return

    _, state_path = _dataloader_save_path(config.checkpoint, resume_step, dp_rank)
    assert state_path and os.path.exists(state_path), (
        f"[energon] dataloader state not found: "
        f"resume_step={resume_step} dp={dp_rank}/{dp_size} path={state_path}"
    )
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    dataloader.restore_state(state)
    if require_pretrain_metadata:
        restored_metadata = dataloader.get_checkpoint_metadata()
        assert restored_metadata, (f"checkpoint {state_path} is missing pretrain progress metadata")
        restored_num_samples = int(restored_metadata["raw_num_samples"])
        assert expected_num_samples is not None
        assert restored_num_samples == expected_num_samples, (
            f"dataset size changed across resume: checkpoint has {restored_num_samples}, "
            f"current dataset has {expected_num_samples}"
        )
        consumed_samples = int(restored_metadata["consumed_samples"])
        assert consumed_samples >= 0, (
            f"invalid consumed_samples={consumed_samples} in {state_path}"
        )
        skipped_samples = int(restored_metadata["skipped_samples"])
        assert 0 <= skipped_samples <= consumed_samples, (
            f"invalid skipped_samples={skipped_samples}, "
            f"consumed_samples={consumed_samples} in {state_path}"
        )
    print(
        f"[energon] restored dataloader state: "
        f"resume_step={resume_step} dp={dp_rank}/{dp_size} path={state_path}",
        flush=True,
    )


class MapFuncCooker:
    """Cook CrudeSample through a user ``map_func`` (or SkipSample)."""

    __stateless__ = True

    def __init__(
        self,
        map_func: Callable,
        sample_process_func: Callable,
        tokens_key: str = "tokens",
        max_seq_length: Optional[int] = None,
        skip_as_marker: bool = False,
    ):
        self.map_func = map_func
        self.sample_process_func = sample_process_func
        self.tokens_key = tokens_key
        self.max_seq_length = max_seq_length
        self.skip_as_marker = skip_as_marker

    def _skip(self, sample: CrudeSample, reason: str) -> RawSample:
        key = sample.get("__key__")
        print(f"[skip] {reason}: {key}")
        if self.skip_as_marker:
            return RawSample(
                **basic_sample_keys(sample),
                data={
                    "is_skipped": True,
                    "key": key
                },
            )
        raise SkipSample()

    @stateless(failure_tolerance=1000)
    def __call__(self, sample: CrudeSample) -> RawSample:
        key = sample.get("__key__")
        try:
            formatted = self.sample_process_func(sample)
            data_dict = self.map_func(formatted)
            if not isinstance(data_dict, dict):
                return self._skip(sample, f"map_func failed -> {data_dict}")

            item = dict(data_dict)
            if self.tokens_key not in item:
                return self._skip(sample, f"missing tokens key: {self.tokens_key}")

            if self.max_seq_length is not None:
                seq_len = int(item[self.tokens_key].shape[-1])
                if seq_len > self.max_seq_length:
                    return self._skip(sample, f"too long ({seq_len}>{self.max_seq_length})")

            if "key" not in item:
                item["key"] = key
            return RawSample(**basic_sample_keys(sample), data=item)
        except SkipSample:
            if self.skip_as_marker:
                return RawSample(
                    **basic_sample_keys(sample),
                    data={
                        "is_skipped": True,
                        "key": key
                    },
                )
            raise
        except Exception as exc:  # noqa: BLE001
            return self._skip(sample, f"cook error -> {exc}")


class MapFuncTaskEncoder(DefaultTaskEncoder):
    decoder = None

    def __init__(self, cooker: MapFuncCooker):
        super().__init__()
        self.cookers = [Cooker(cook=cooker)]
        self.decoder = None


class CollatorWrapper:
    __stateless__ = True

    def __init__(self, collate_fn: Callable):
        self.collate_fn = collate_fn

    @stateless()
    def __call__(self, samples: List[RawSample]) -> BatchedSample:
        batch = self.collate_fn([s.data for s in samples])
        return BatchedSample(__key__=None, __restore_key__=None, data=batch)


class EnergonDataloader:
    """Wrap savable energon loader so per-epoch ``iter()`` does not rewind."""
    def __init__(
        self,
        loader,
        length: int,
        checkpoint_metadata: Optional[Dict[str, Any]] = None,
    ):
        self._loader = loader
        self._length = max(int(length), 1)
        self._checkpoint_metadata = dict(checkpoint_metadata or {})
        self._iter = None

    def __iter__(self):
        if self._iter is None:
            self._iter = iter(self._loader)
        return self

    def __next__(self):
        if self._iter is None:
            self._iter = iter(self._loader)
        batch = next(self._iter)
        if hasattr(batch, "data"):
            return batch.data
        return batch

    def __len__(self):
        return self._length

    def save_state(self):
        return {
            "loader_state": self._loader.save_state_rank(),
            "checkpoint_metadata": self.get_checkpoint_metadata(),
        }

    def get_checkpoint_metadata(self) -> Dict[str, Any]:
        return dict(self._checkpoint_metadata)

    def update_checkpoint_metadata(self, **metadata: Any) -> None:
        self._checkpoint_metadata.update(metadata)

    def restore_state(self, state):
        normalized = normalize_energon_ckpt_state(state)
        self._checkpoint_metadata = dict(normalized["checkpoint_metadata"])
        self._loader.restore_state_rank(normalized["loader_state"])
        self._iter = None


class CarryOverPackingDataset(PackingDataset):
    """Like ``PackingDataset``, but carries the last under-full bin mid-stream."""

    _savable_fields = PackingDataset._savable_fields + ("_carry_over", )

    def __init__(
        self,
        dataset: SavableDataset,
        buffer_size: int,
        pre_packer: Callable[[List[T_sample]], List[List[T_sample]]],
        final_packer: Callable[[List[T_encoded_sample]], T_batch_sample],
        *,
        pack_bin_size: int,
        sample_length_fn: Callable[[T_sample], int],
        final_packer_stateless: bool = False,
        sample_encoder: Optional[Callable[[T_sample], T_encoded_sample]] = None,
        sample_encoder_stateless: bool = False,
        packer_config: Optional[Union[Dict[str, Any], Callable[[], Dict[str, Any]]]] = None,
        pre_packer_failure_tolerance: int = 100,
        final_packer_failure_tolerance: int = 100,
        sample_encoder_failure_tolerance: int = 100,
        worker_config: WorkerConfig,
    ):
        self._inner_pre_packer = pre_packer
        self._pack_bin_size = int(pack_bin_size)
        self._sample_length_fn = sample_length_fn
        super().__init__(
            dataset,
            buffer_size,
            pre_packer=self._pre_pack_with_carry,
            final_packer=final_packer,
            final_packer_stateless=final_packer_stateless,
            sample_encoder=sample_encoder,
            sample_encoder_stateless=sample_encoder_stateless,
            packer_config=packer_config,
            pre_packer_failure_tolerance=pre_packer_failure_tolerance,
            final_packer_failure_tolerance=final_packer_failure_tolerance,
            sample_encoder_failure_tolerance=sample_encoder_failure_tolerance,
            worker_config=worker_config,
        )

    def reset_state_own(self) -> None:
        super().reset_state_own()
        self._carry_over = SavableSampleBuffer(self.dataset, worker_config=self.worker_config)

    def _pre_pack_with_carry(self, samples: List[T_sample]) -> List[List[T_sample]]:
        self._carry_over.worker_start()
        combined = list(self._carry_over.buffer) + list(samples)
        self._carry_over.clear()
        if not combined:
            return []
        packs = self._inner_pre_packer(combined)
        if not packs:
            return []
        n_out = sum(len(p) for p in packs)
        assert n_out == len(combined), (
            f"pre_packer must return a partition of inputs; "
            f"got {n_out} samples from {len(combined)}"
        )
        last = packs[-1]
        last_len = sum(self._sample_length_fn(s) for s in last)
        if last_len < self._pack_bin_size:
            for s in last:
                self._carry_over.append(s)
            return packs[:-1]
        return packs

    def config(self) -> Dict[str, Any]:
        cfg = super().config()
        cfg["type"] = type(self).__qualname__
        cfg["pack_bin_size"] = self._pack_bin_size
        cfg["inner_pre_packer"] = self._function_config(self._inner_pre_packer)
        return cfg


def round_up(n: int, divisor: int) -> int:
    if divisor <= 1:
        return n
    return ((n + divisor - 1) // divisor) * divisor


def compute_pack_align(tp_size: int, cp_size: int) -> int:
    cp_pad = 2 * cp_size if cp_size > 1 else 1
    tp_pad = tp_size if tp_size > 1 else 1
    return int(cp_pad * tp_pad)


def greedy_next_fit_bin_pack(
    sample_lengths: List[int],
    bin_size: int,
) -> List[List[int]]:
    packs: List[List[int]] = []
    current: List[int] = []
    current_len = 0
    for idx, length in enumerate(sample_lengths):
        assert length <= bin_size, (
            f"padded sample length {length} exceeds pack_bin_size {bin_size}"
        )
        if current and current_len + length > bin_size:
            packs.append(current)
            current = [idx]
            current_len = length
        else:
            current.append(idx)
            current_len += length
    if current:
        packs.append(current)
    return packs


def estimate_packed_epoch_length(
    num_samples: int,
    dp_size: int,
    batch_size: int,
    est_docs_per_pack: int,
) -> int:
    assert est_docs_per_pack >= 1
    denom = max(dp_size * batch_size * est_docs_per_pack, 1)
    return max(num_samples // denom, 1)


def resolve_est_docs_per_pack(task, pack_bin_size: int) -> int:
    """Resolve mean docs/pack for epoch-length estimation.

    Prefer ``task.est_avg_seqlen`` (from metrics: tokens_sum / samples_sum);
    fall back to ``task.est_docs_per_pack``.
    """
    if "est_avg_seqlen" in task and task["est_avg_seqlen"] is not None:
        avg_seqlen = int(task["est_avg_seqlen"])
        assert avg_seqlen >= 1, f"est_avg_seqlen must be >= 1, got {avg_seqlen}"
        return max(int(pack_bin_size) // avg_seqlen, 1)
    assert "est_docs_per_pack" in task, (
        "task.est_avg_seqlen (preferred) or task.est_docs_per_pack is required "
        "when packing (used only for epoch length)"
    )
    est_docs_per_pack = int(task["est_docs_per_pack"])
    assert est_docs_per_pack >= 1, (f"est_docs_per_pack must be >= 1, got {est_docs_per_pack}")
    return est_docs_per_pack


class PackingTaskEncoder(MapFuncTaskEncoder):
    """Cook + sequential greedy next-fit pack into THD microbatches."""
    def __init__(
        self,
        cooker,
        collate_func: Callable,
        pack_bin_size: int,
        align: int,
        tokens_key: str = "tokens",
    ):
        super().__init__(cooker)
        self.collate_func = collate_func
        self.tokens_key = tokens_key
        self.pack_bin_size = int(pack_bin_size)
        self.align = int(align)
        assert self.pack_bin_size % self.align == 0, (
            f"pack_bin_size={self.pack_bin_size} must be divisible by align={self.align}"
        )

    def _padded_len(self, sample: RawSample) -> int:
        if sample.data.get("is_skipped", False):
            return 0
        return round_up(int(sample.data[self.tokens_key].shape[-1]), self.align)

    # 选多少个样本进行打包
    def select_samples_to_pack(self, samples: List[RawSample]) -> List[List[RawSample]]:
        lengths = [self._padded_len(s) for s in samples]
        index_packs = greedy_next_fit_bin_pack(lengths, self.pack_bin_size)
        return [[samples[i] for i in idxs] for idxs in index_packs]

    def build_batch(
        self,
        dataset: SavableDataset,
        *,
        batch_size: Optional[int],
        batch_drop_last: bool = False,
        packing_buffer_size: Optional[int] = None,
        worker_config: WorkerConfig,
    ) -> SavableDataset:
        assert packing_buffer_size is not None, ("PackingTaskEncoder requires packing_buffer_size")
        dataset = CarryOverPackingDataset(
            dataset,
            buffer_size=packing_buffer_size,
            pre_packer=self.select_samples_to_pack,
            final_packer=self.pack_selected_samples,
            pack_bin_size=self.pack_bin_size,
            sample_length_fn=self._padded_len,
            final_packer_stateless=get_stateless(self.pack_selected_samples),
            sample_encoder=None,
            sample_encoder_stateless=True,
            worker_config=worker_config,
            pre_packer_failure_tolerance=get_failure_tolerance(
                self.select_samples_to_pack, self.__default_failure_tolerance__
            ),
            final_packer_failure_tolerance=get_failure_tolerance(
                self.pack_selected_samples, self.__default_failure_tolerance__
            ),
            sample_encoder_failure_tolerance=0,
        )
        assert batch_size is not None and batch_size > 0
        dataset = BatchDataset(
            dataset,
            batch_size=batch_size,
            batcher=self.batch,
            drop_last=batch_drop_last,
            worker_config=worker_config,
            batcher_stateless=get_stateless(self.batch),
            failure_tolerance=get_failure_tolerance(self.batch, self.__default_failure_tolerance__),
        )
        if self._is_overridden(self.encode_batch):
            dataset = MapDataset(
                dataset,
                self.encode_batch,
                worker_config=worker_config,
                stateless_map_fn=get_stateless(self.encode_batch),
                failure_tolerance=get_failure_tolerance(
                    self.encode_batch, self.__default_failure_tolerance__
                ),
            )
        return dataset

    @stateless(failure_tolerance=100)
    def pack_selected_samples(self, samples: List[RawSample]) -> RawSample:
        assert samples, "empty pack"
        num_skipped_samples = sum(int(sample.data.get("is_skipped", False)) for sample in samples)
        samples = [sample for sample in samples if not sample.data.get("is_skipped", False)]
        if not samples:
            raise FatalSampleError("skip-only packs must remain in carry-over")

        packed = self.collate_func(
            [sample.data for sample in samples],
            align=self.align,
            pack_bin_size=self.pack_bin_size,
        )
        assert isinstance(packed,
                          dict), (f"packed collate_func must return dict, got {type(packed)}")
        assert "num_skipped_samples" not in packed, (
            f"packed collate_func must not contain num_skipped_samples, got {packed}"
        )
        packed["num_skipped_samples"] = num_skipped_samples
        return RawSample.extend(
            samples[0],
            __key__=",".join(str(s.__key__) for s in samples),
            data=packed,
        )

    @stateless()
    def batch(self, samples: List[RawSample]) -> BatchedSample:
        batch = collate_fn_packed([s.data for s in samples])
        return BatchedSample(__key__=None, __restore_key__=None, data=batch)


def _task_mapping(config) -> Any:
    task = config.task
    if task is None:
        return {}
    return task


def _shuffle_buffer_size(task) -> Optional[int]:
    if "shuffle_buffer_size" not in task:
        return None
    return int(task["shuffle_buffer_size"])


def _parallel_shard_iters(task) -> int:
    if "parallel_shard_iters" in task:
        return int(task["parallel_shard_iters"])
    return 16


def _packing_enabled(task) -> bool:
    return "pack_bin_size" in task and task["pack_bin_size"] is not None


def get_packed_energon_dataset_and_dataloader(
    config,
    map_func: Callable,
    sample_process_func: Callable,
    collate_func: Callable,
    tokens_key: str = "tokens",
    dp_rank: int = 0,
    dp_size: int = 1,
    meta_info: Optional[Dict[str, Any]] = None,
    data_path: Optional[str] = None,
    max_seq_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a dataset-packed THD Energon dataloader."""
    assert config is not None
    assert config.data.data_pathes, (
        "data.data_pathes must point to a WebDataset or metadataset.yaml"
    )
    assert map_func is not None, "map_func is required"
    task = _task_mapping(config)
    assert _packing_enabled(task), ("task.pack_bin_size is required by the packed Energon builder")
    path_in = data_path if data_path is not None else config.data.data_pathes[0]
    data_path = resolve_energon_data_path(path_in)
    batch_size = config.training.train_mbs
    max_seq_length = (config.training.seq_length if max_seq_length is None else max_seq_length)
    prefetch = config.data.dataloader_prefetch_factor
    shuffle_buffer_size = _shuffle_buffer_size(task)
    parallel_shard_iters = _parallel_shard_iters(task)
    worker_config = WorkerConfig(
        rank=dp_rank,
        world_size=dp_size,
        num_workers=config.data.dataloader_num_workers,
        seed_offset=config.data.sampler_seed,
    )
    cooker = MapFuncCooker(
        map_func,
        sample_process_func=sample_process_func,
        tokens_key=tokens_key,
        max_seq_length=max_seq_length,
        skip_as_marker=True,
    )

    dist_config = config.policy.dist_config
    assert not dist_config.dynamic_context_parallel, (
        "energon packing is incompatible with dynamic_context_parallel; "
        "disable dyn-CP in the yaml"
    )
    assert batch_size == 1, ("dataset packing requires training.train_mbs=1 (THD microbatch)")

    pack_bin_size = int(task["pack_bin_size"])
    packing_buffer_size = (
        int(task["packing_buffer_size"]) if "packing_buffer_size" in task else 64
    )
    tp_size = int(dist_config.tensor_model_parallel_size)
    cp_size = int(dist_config.context_parallel_size)
    align = (
        int(task["pack_align"]) if "pack_align" in task else compute_pack_align(tp_size, cp_size)
    )
    assert pack_bin_size % align == 0, (
        f"pack_bin_size={pack_bin_size} must be divisible by align={align}"
    )
    assert pack_bin_size >= round_up(max_seq_length, align), (
        f"pack_bin_size={pack_bin_size} must be >= round_up(seq_length="
        f"{max_seq_length}, align={align})"
    )

    print(
        f"[energon] loading {data_path} packing "
        f"pack_bin_size={pack_bin_size} align={align} buffer={packing_buffer_size}",
        flush=True,
    )
    task_encoder = PackingTaskEncoder(
        cooker,
        collate_func=collate_func,
        tokens_key=tokens_key,
        pack_bin_size=pack_bin_size,
        align=align,
    )
    train_dataset = get_train_dataset(
        data_path,
        worker_config=worker_config,
        batch_size=batch_size,
        batch_drop_last=True,
        packing_buffer_size=packing_buffer_size,
        shuffle_buffer_size=shuffle_buffer_size,
        max_samples_per_sequence=None,
        parallel_shard_iters=parallel_shard_iters,
        task_encoder=task_encoder,
    )
    setattr(train_dataset, "gcore_pack_bin_size", pack_bin_size)

    num_samples = estimate_num_samples(data_path)
    est_docs_per_pack = resolve_est_docs_per_pack(task, pack_bin_size)
    length = estimate_packed_epoch_length(
        num_samples,
        dp_size,
        batch_size,
        est_docs_per_pack,
    )
    checkpoint_metadata = {
        "raw_num_samples": num_samples,
        "consumed_samples": 0,
        "skipped_samples": 0,
    }
    avg_note = (
        f"est_avg_seqlen={task['est_avg_seqlen']} "
        if "est_avg_seqlen" in task and task["est_avg_seqlen"] is not None else ""
    )
    print(
        f"[energon] raw_samples={num_samples} packed_epoch_length(mbs/dp)={length} "
        f"{avg_note}est_docs_per_pack={est_docs_per_pack}",
        flush=True,
    )

    assert not hasattr(train_dataset, "gcore_map_func")
    setattr(train_dataset, "gcore_map_func", map_func)
    loader = get_savable_loader(
        train_dataset,
        watchdog_timeout_seconds=None,
        prefetch_factor=prefetch if prefetch is not None else 2,
    )
    dataloader = EnergonDataloader(
        loader,
        length=length,
        checkpoint_metadata=checkpoint_metadata,
    )
    maybe_restore_dataloader(
        dataloader,
        config,
        dp_rank,
        dp_size,
        meta_info,
        require_pretrain_metadata=True,
        expected_num_samples=num_samples,
    )
    return {
        "train_dataset": train_dataset,
        "train_sampler": None,
        "train_dataloader": dataloader,
    }


def get_non_packed_energon_dataset_and_dataloader(
    config,
    map_func: Callable,
    sample_process_func: Callable,
    tokens_key: str = "tokens",
    dp_rank: int = 0,
    dp_size: int = 1,
    meta_info: Optional[Dict[str, Any]] = None,
    collate_func: Optional[Callable] = None,
    data_path: Optional[str] = None,
    max_seq_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a regular batched Energon dataloader without dataset packing."""
    assert config is not None
    assert config.data.data_pathes, (
        "data.data_pathes must point to a WebDataset or metadataset.yaml"
    )
    assert map_func is not None, "map_func is required"
    task = _task_mapping(config)
    path_in = data_path if data_path is not None else config.data.data_pathes[0]
    data_path = resolve_energon_data_path(path_in)
    batch_size = config.training.train_mbs
    max_seq_length = (config.training.seq_length if max_seq_length is None else max_seq_length)
    prefetch = config.data.dataloader_prefetch_factor
    shuffle_buffer_size = _shuffle_buffer_size(task)
    parallel_shard_iters = _parallel_shard_iters(task)
    worker_config = WorkerConfig(
        rank=dp_rank,
        world_size=dp_size,
        num_workers=config.data.dataloader_num_workers,
        seed_offset=config.data.sampler_seed,
    )
    cooker = MapFuncCooker(
        map_func,
        sample_process_func=sample_process_func,
        tokens_key=tokens_key,
        max_seq_length=max_seq_length,
        skip_as_marker=False,
    )
    print(f"[energon] loading {data_path}", flush=True)
    task_encoder = MapFuncTaskEncoder(cooker)
    train_dataset = get_train_dataset(
        data_path,
        worker_config=worker_config,
        batch_size=None,
        shuffle_buffer_size=shuffle_buffer_size,
        max_samples_per_sequence=None,
        parallel_shard_iters=parallel_shard_iters,
        task_encoder=task_encoder,
    )
    train_dataset = BatchDataset(
        train_dataset,
        batch_size=batch_size,
        batcher=CollatorWrapper(collate_func),
        batcher_stateless=True,
        drop_last=True,
        worker_config=worker_config,
    )

    num_samples = estimate_num_samples(data_path)
    length = max(
        num_samples // max(dp_size * batch_size, 1),
        1,
    )
    print(
        f"[energon] dp={dp_rank}/{dp_size} "
        f"estimated samples={num_samples} length={length}",
        flush=True,
    )

    assert not hasattr(train_dataset, "gcore_map_func")
    setattr(train_dataset, "gcore_map_func", map_func)
    loader = get_savable_loader(
        train_dataset,
        watchdog_timeout_seconds=None,
        prefetch_factor=prefetch if prefetch is not None else 2,
    )
    dataloader = EnergonDataloader(
        loader,
        length=length,
        checkpoint_metadata={},
    )
    maybe_restore_dataloader(
        dataloader,
        config,
        dp_rank,
        dp_size,
        meta_info,
        require_pretrain_metadata=False,
        expected_num_samples=None,
    )
    return {
        "train_dataset": train_dataset,
        "train_sampler": None,
        "train_dataloader": dataloader,
    }


def verify_dataloader_func(train_dataset, train_sampler, train_dataloader):
    print(f"{len(train_dataset)=}")
    print(f"{len(train_dataloader)=}")
    batch = next(iter(train_dataloader))
    print(f"batch keys: {batch.keys()}")
    tokens = batch["tokens"]
    if isinstance(tokens, list):
        tokens = tokens[0]
        labels = batch["labels"][0]
    else:
        labels = batch["labels"]
    tokenizer = train_dataset.gcore_map_func.tokenizer
    kept = [t for t, l in zip(tokens.tolist(), labels.tolist()) if l != -100]
    print(f"kept decoded (trunc):\n{tokenizer.decode(kept[:256])}")
    if "cu_seqlens_padded" in batch:
        print(
            f"packed tokens={tokens.shape} labels={labels.shape} "
            f"cu_seqlens_padded={batch['cu_seqlens_padded'].tolist()} "
            f"num_docs={batch['num_docs']}"
        )
