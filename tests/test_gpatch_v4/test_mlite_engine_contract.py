import inspect
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


MLITE_PATH = Path(__file__).resolve().parents[3] / "mlite" / "experimental" / "lite"
sys.path.insert(0, str(MLITE_PATH))

import gpatch_v4.training_backend as training_backend  # noqa: E402
import gpatch_v4.training_backend.mlite_backend.checkpoint as mlite_checkpoint_module  # noqa: E402
import gpatch_v4.training_backend.mlite_backend.config as mlite_config_module  # noqa: E402
import gpatch_v4.training_backend.mlite_backend.mlite_engine as mlite_engine_module  # noqa: E402
from gpatch_v4.configs import FinetuneConfig, RlConfig  # noqa: E402
from gpatch_v4.training_backend.mlite_backend import MliteEngine  # noqa: E402


def _finetune_config() -> FinetuneConfig:
    config = FinetuneConfig()
    config.training.training_backend = "mlite"
    config.training.loss_func = "cross_entropy"
    config.training.build_from_mbridge = False
    config.training.total_training_step = 10
    config.policy.model_arch = "qwen3_5_moe"
    config.policy.without_ref = True
    config.policy.hf_model_path = "/tmp/qwen3.5"
    config.policy.hf_tokenizer_path = "/tmp/qwen3.5"
    config.policy.override_transformer_config = {}
    config.policy.dist_config.tensor_model_parallel_size = 1
    config.policy.dist_config.expert_tensor_parallel_size = 1
    config.policy.dist_config.dynamic_context_parallel = False
    config.checkpoint.use_dist_checkpointing = True
    return config


def _patch_hf_resolve(monkeypatch, hf_config=None) -> None:
    if hf_config is None:
        hf_config = SimpleNamespace()
    monkeypatch.setattr(
        mlite_engine_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: hf_config,
    )
    monkeypatch.setattr(
        mlite_config_module,
        "resolve_model_type_from_hf",
        lambda _config: "qwen3_5",
    )


def _engine(monkeypatch) -> MliteEngine:
    _patch_hf_resolve(monkeypatch)
    config = _finetune_config()
    return MliteEngine(config, config.policy, None)


@pytest.mark.parametrize("model_arch", ["qwen3_5", "qwen3_5_moe"])
def test_mlite_config_maps_fsdp2_thd_and_parallel_dimensions(monkeypatch, model_arch):
    config = _finetune_config()
    config.policy.model_arch = model_arch
    _patch_hf_resolve(monkeypatch)
    engine = MliteEngine(config, config.policy, None)

    mlite_config = engine._build_mlite_config()

    assert mlite_config.impl_cfg["optimizer"] == "fsdp2"
    assert mlite_config.impl_cfg["use_thd"] is True
    assert mlite_config.parallel.tp == 1
    assert mlite_config.parallel.etp == 1
    assert "runtime_plugins" not in mlite_config.impl_cfg


def test_mlite_dense_config_requires_ep_one(monkeypatch):
    config = _finetune_config()
    config.policy.model_arch = "qwen3_5"
    config.policy.dist_config.expert_model_parallel_size = 2
    _patch_hf_resolve(monkeypatch)

    with pytest.raises(NotImplementedError, match="qwen3_5 dense requires ep=1"):
        MliteEngine(config, config.policy, None)


def test_mlite_config_allows_online_hf_export_and_caches_metadata(monkeypatch):
    config = _finetune_config()
    config.checkpoint.convert_mcore_to_hf_online = True
    config.checkpoint.save_ckpt_path = "/tmp/mlite"
    _patch_hf_resolve(monkeypatch)
    cache_calls = []
    monkeypatch.setattr(
        mlite_engine_module,
        "cache_hf_metadata_files",
        lambda *args: cache_calls.append(args),
    )

    MliteEngine(config, config.policy, None)

    assert cache_calls == [("/tmp/qwen3.5", "/tmp/mlite")]


@pytest.mark.parametrize(
    "field",
    ["convert_mcore_to_hf_offline", "skip_save_mcore_model"],
)
def test_mlite_config_rejects_unsupported_checkpoint_modes(monkeypatch, field):
    config = _finetune_config()
    setattr(config.checkpoint, field, True)
    _patch_hf_resolve(monkeypatch)

    with pytest.raises(NotImplementedError, match="does not support offline HF conversion"):
        MliteEngine(config, config.policy, None)


def test_mlite_config_maps_vision_mount_and_freeze(monkeypatch):
    config = _finetune_config()
    config.policy.model_arch = "qwen3_5"
    config.training.freeze_vit = True
    config.training.freeze_projector = True
    _patch_hf_resolve(
        monkeypatch,
        SimpleNamespace(
            model_type="qwen3_5",
            vision_config=SimpleNamespace(),
        ),
    )
    engine = MliteEngine(config, config.policy, None)

    mlite_config = engine._build_mlite_config()

    assert mlite_config.impl_cfg["mount_vision_model"] is True
    assert mlite_config.impl_cfg["freeze_vision_model"] is True
    assert mlite_config.impl_cfg["freeze_vision_projector"] is True


def test_mlite_vision_mount_follows_hf_vision_config(monkeypatch):
    config = _finetune_config()
    config.policy.model_arch = "qwen3_5"
    _patch_hf_resolve(monkeypatch, SimpleNamespace(model_type="qwen3_5"))
    engine = MliteEngine(config, config.policy, None)

    mlite_config = engine._build_mlite_config()

    assert mlite_config.impl_cfg["mount_vision_model"] is False


@pytest.mark.parametrize(
    ("model_arch", "hf_model_type"),
    [
        ("qwen3_5", "qwen3_5_moe"),
        ("qwen3_5_moe", "qwen3_5"),
        ("qwen3_5", "welmv4_moe"),
        ("qwen3_5_moe", "welmv4_moe"),
    ],
)
def test_mlite_config_rejects_policy_hf_variant_mismatch(
    monkeypatch,
    model_arch,
    hf_model_type,
):
    config = _finetune_config()
    config.policy.model_arch = model_arch
    _patch_hf_resolve(monkeypatch, SimpleNamespace(model_type=hf_model_type))
    engine = MliteEngine(config, config.policy, None)

    with pytest.raises(ValueError, match="does not match HF model_type"):
        engine._build_mlite_config()


def test_mlite_vision_mount_maps_dynamic_cp_plugin(monkeypatch):
    config = _finetune_config()
    config.policy.model_arch = "qwen3_5"
    config.policy.dist_config.dynamic_context_parallel = True
    config.policy.dist_config.max_seqlen_per_dp_cp_rank = 4096
    _patch_hf_resolve(
        monkeypatch,
        SimpleNamespace(
            model_type="qwen3_5",
            vision_config=SimpleNamespace(),
        ),
    )

    engine = MliteEngine(config, config.policy, None)
    mlite_config = engine._build_mlite_config()

    assert mlite_config.impl_cfg["mount_vision_model"] is True
    assert mlite_config.impl_cfg["runtime_plugins"]["dynamic_context_parallel"]["enabled"] is True


def test_mlite_config_maps_dynamic_cp_runtime_plugin(monkeypatch):
    config = _finetune_config()
    config.training.attention_backend = "flash"
    config.policy.dist_config.dynamic_context_parallel = True
    config.policy.dist_config.max_seqlen_per_dp_cp_rank = 4096
    config.policy.dist_config.min_dynamic_context_parallel_size = 1
    config.policy.dist_config.context_parallel_size = 1
    _patch_hf_resolve(monkeypatch)
    engine = MliteEngine(config, config.policy, None)

    mlite_config = engine._build_mlite_config()
    plugin = mlite_config.impl_cfg["runtime_plugins"]["dynamic_context_parallel"]

    assert plugin["enabled"] is True
    assert plugin["max_seqlen_per_dp_cp_rank"] == 4096
    assert plugin["min_context_parallel_size"] == 1
    assert plugin["require_full_cp_size_coverage"] is False


@pytest.mark.parametrize("attention_backend", ["auto", "fused"])
def test_dynamic_cp_non_welm_mlite_requires_flash(attention_backend):
    from gpatch_v4.configs.config import _assert_dynamic_cp_requires

    training = SimpleNamespace(
        training_backend="mlite",
        attention_backend=attention_backend,
    )
    policy = SimpleNamespace(
        model_arch="qwen3_5",
        dist_config=SimpleNamespace(
            dynamic_context_parallel=True,
            dynamic_cp_scheduler_type="default",
            context_parallel_size=1,
        ),
    )

    with pytest.raises(
        AssertionError,
        match="dynamic_context_parallel requires training.attention_backend='flash'",
    ):
        _assert_dynamic_cp_requires(training, policy)


def test_mlite_config_rejects_non_default_dynamic_cp_scheduler(monkeypatch):
    config = _finetune_config()
    config.training.attention_backend = "flash"
    config.policy.dist_config.dynamic_context_parallel = True
    config.policy.dist_config.max_seqlen_per_dp_cp_rank = 4096
    config.policy.dist_config.dynamic_cp_scheduler_type = "smart_padding"
    _patch_hf_resolve(monkeypatch)

    with pytest.raises(NotImplementedError, match="dynamic_cp_scheduler_type='default'"):
        MliteEngine(config, config.policy, None)


def test_mlite_factory_fails_fast_when_dependency_is_missing(monkeypatch):
    config = _finetune_config()
    monkeypatch.setattr(training_backend, "MliteEngine", None)
    monkeypatch.setattr(
        training_backend,
        "_MLITE_IMPORT_ERROR",
        ImportError("missing megatron.lite"),
    )

    with pytest.raises(ImportError, match="put mlite/experimental/lite before"):
        training_backend.TrainingEngineFactory.get_training_engine(config)


def test_mlite_factory_rejects_rl_config_before_engine_construction():
    config = RlConfig()
    config.training.training_backend = "mlite"

    with pytest.raises(NotImplementedError, match="FinetuneConfig only"):
        training_backend.TrainingEngineFactory.get_training_engine(config)


def test_checkpoint_signature_marker_scheduler_and_resume(monkeypatch, tmp_path):
    engine = _engine(monkeypatch)
    engine.checkpoint_config.save_ckpt_path = str(tmp_path)

    class Scheduler:
        def __init__(self):
            self.loaded_state = None

        def state_dict(self):
            return {"num_steps": 7}

        def load_state_dict(self, state):
            self.loaded_state = state

    class Runtime:
        def __init__(self):
            self.save_kwargs = None
            self.load_kwargs = None

        def save_checkpoint(self, _handle, path, *, step, **kwargs):
            self.save_kwargs = kwargs
            os.makedirs(os.path.join(path, f"step_{step}"), exist_ok=True)

        def load_checkpoint(self, _handle, path, **kwargs):
            del _handle, path
            self.load_kwargs = kwargs
            return 7

    scheduler = Scheduler()
    runtime = Runtime()
    engine.runtime = runtime
    engine.handle = SimpleNamespace(_lr_scheduler=scheduler)

    parameters = list(inspect.signature(engine.save_checkpoint).parameters)
    assert parameters == ["global_step", "dataloader"]
    engine.save_checkpoint(7)

    marker = tmp_path / "latest_checkpointed_iteration.txt"
    assert marker.read_text(encoding="utf-8") == "7"
    assert (tmp_path / "step_7" / "lr_scheduler.pt").is_file()
    assert runtime.save_kwargs["save_optimizer"] is True
    assert engine.load_checkpoint(str(tmp_path)) == 7
    assert runtime.load_kwargs["load_optimizer"] is True
    assert scheduler.loaded_state == {"num_steps": 7}


def _online_checkpoint_engine(monkeypatch, tmp_path):
    engine = _engine(monkeypatch)
    engine.checkpoint_config.save_ckpt_path = str(tmp_path)
    engine.checkpoint_config.convert_mcore_to_hf_online = True
    calls = SimpleNamespace(dcp=[], hf=[], metadata=[])

    class Runtime:
        def save_checkpoint(self, _handle, path, *, step, **kwargs):
            del _handle
            calls.dcp.append((path, step, kwargs))
            os.makedirs(os.path.join(path, f"step_{step}"), exist_ok=True)

    def save_hf_weights(model_chunks, path, model_config, parallel_state):
        calls.hf.append((model_chunks, path, model_config, parallel_state))

    model_chunks = [object()]
    model_config = object()
    parallel_state = object()
    engine.runtime = Runtime()
    engine.handle = SimpleNamespace(
        _lr_scheduler=SimpleNamespace(state_dict=lambda: {"num_steps": 7}),
        _parallel_state=parallel_state,
        _extras={
            "protocol": SimpleNamespace(save_hf_weights=save_hf_weights),
            "model_chunks": model_chunks,
            "model_cfg": model_config,
        },
    )
    monkeypatch.setattr(
        mlite_checkpoint_module,
        "copy_cached_hf_metadata_files",
        lambda *args: calls.metadata.append(args),
    )
    monkeypatch.setattr(mlite_checkpoint_module, "log", lambda *args, **kwargs: None)
    return engine, calls, model_chunks, model_config, parallel_state


@pytest.mark.parametrize("custom_export_root", [False, True])
def test_online_hf_checkpoint_export_path_and_metadata(
    monkeypatch,
    tmp_path,
    custom_export_root,
):
    engine, calls, model_chunks, model_config, parallel_state = _online_checkpoint_engine(
        monkeypatch,
        tmp_path,
    )
    if custom_export_root:
        export_root = tmp_path / "export"
        engine.checkpoint_config.export_hf_save_path = str(export_root)
    else:
        export_root = tmp_path / "hf"

    engine.save_checkpoint(7)

    hf_path = str(export_root / "7")
    assert calls.dcp == [
        (
            str(tmp_path),
            7,
            {"use_dcp": True, "save_optimizer": True},
        )
    ]
    assert calls.hf == [(model_chunks, hf_path, model_config, parallel_state)]
    assert calls.metadata == [
        ("/tmp/qwen3.5", str(tmp_path), hf_path),
    ]
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "7"


def test_online_hf_checkpoint_export_runs_on_nonzero_rank(monkeypatch, tmp_path):
    engine, calls, _, _, _ = _online_checkpoint_engine(monkeypatch, tmp_path)
    monkeypatch.setattr(mlite_checkpoint_module, "_rank", lambda: 1)

    engine.save_checkpoint(7)

    assert len(calls.hf) == 1
    assert not (tmp_path / "latest_checkpointed_iteration.txt").exists()


def test_online_hf_checkpoint_export_requires_protocol_hook(monkeypatch, tmp_path):
    engine, _, _, _, _ = _online_checkpoint_engine(monkeypatch, tmp_path)
    engine.handle._extras["protocol"] = SimpleNamespace()

    with pytest.raises(RuntimeError, match="protocol to expose save_hf_weights"):
        engine.save_checkpoint(7)

    assert not (tmp_path / "latest_checkpointed_iteration.txt").exists()


@pytest.mark.parametrize("rank", [0, 1])
def test_online_hf_checkpoint_metadata_failure_is_collective(monkeypatch, tmp_path, rank):
    engine, _, _, _, _ = _online_checkpoint_engine(monkeypatch, tmp_path)
    broadcasts = []
    monkeypatch.setattr(mlite_checkpoint_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(mlite_checkpoint_module, "_rank", lambda: rank)

    def copy_metadata(*args):
        del args
        if rank == 0:
            raise OSError("missing config")
        pytest.fail("nonzero rank must not copy HF metadata")

    def broadcast_error(payload, src):
        assert src == 0
        if rank == 1:
            payload[0] = "OSError: missing config"
        broadcasts.append(payload[0])

    monkeypatch.setattr(
        mlite_checkpoint_module,
        "copy_cached_hf_metadata_files",
        copy_metadata,
    )
    monkeypatch.setattr(
        mlite_checkpoint_module.dist,
        "broadcast_object_list",
        broadcast_error,
    )
    monkeypatch.setattr(
        mlite_checkpoint_module.dist,
        "barrier",
        lambda: pytest.fail("metadata failure must not enter the final barrier"),
    )

    with pytest.raises(RuntimeError, match="OSError: missing config"):
        engine.save_checkpoint(7)

    assert broadcasts == ["OSError: missing config"]
    assert not (tmp_path / "latest_checkpointed_iteration.txt").exists()
