import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

MLITE_PATH = Path(__file__).resolve().parents[3] / "mlite" / "experimental" / "lite"
sys.path.insert(0, str(MLITE_PATH))

import gpatch_v4.training_backend as training_backend  # noqa: E402
import gpatch_v4.training_backend.mlite_backend.checkpoint as mlite_checkpoint_module  # noqa: E402
import gpatch_v4.training_backend.mlite_backend.config as mlite_config_module  # noqa: E402
import gpatch_v4.training_backend.mlite_backend.mlite_engine as mlite_engine_module  # noqa: E402
from gpatch_v4.configs import FinetuneConfig  # noqa: E402
from gpatch_v4.training_backend.mlite_backend import MliteEngine  # noqa: E402


def _finetune_config() -> FinetuneConfig:
    config = FinetuneConfig()
    config.training.training_backend = "mlite"
    config.training.loss_func = "cross_entropy"
    config.training.total_training_step = 10
    config.training.freeze_moe_router = False
    config.training.freeze_moe_shared_experts = False
    config.policy.model_arch = "welmv4_moe"
    config.policy.without_ref = True
    config.policy.without_optim = False
    config.policy.wrap_with_ddp = False
    config.policy.post_wrap_with_ddp = False
    config.policy.hf_model_path = "/tmp/welm-v4.5"
    config.policy.hf_tokenizer_path = "/tmp/welm-v4.5"
    config.policy.override_transformer_config = {}
    config.policy.dist_config.tensor_model_parallel_size = 1
    config.policy.dist_config.pipeline_model_parallel_size = 1
    config.policy.dist_config.expert_model_parallel_size = 8
    config.policy.dist_config.expert_tensor_parallel_size = 1
    config.policy.dist_config.context_parallel_size = 1
    config.policy.dist_config.dynamic_context_parallel = False
    config.policy.dist_config.nnodes = 1
    config.checkpoint.use_dist_checkpointing = False
    config.checkpoint.convert_mcore_to_hf_online = False
    return config


def _patch_hf_resolve(monkeypatch, model_type: str = "welmv4_moe") -> None:
    hf_config = SimpleNamespace(model_type=model_type)
    monkeypatch.setattr(
        mlite_engine_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: hf_config,
    )
    monkeypatch.setattr(
        mlite_config_module,
        "resolve_model_type_from_hf",
        lambda _config: "welm_v4_5",
    )


def _engine(monkeypatch) -> MliteEngine:
    _patch_hf_resolve(monkeypatch)
    config = _finetune_config()
    return MliteEngine(config, config.policy, None)


def test_welm_mlite_config_maps_fsdp2_thd_and_parallel_dimensions(monkeypatch):
    engine = _engine(monkeypatch)

    mlite_config = engine._build_mlite_config()

    assert mlite_config.model_name == "welm_v4_5"
    assert mlite_config.impl_cfg["optimizer"] == "fsdp2"
    assert mlite_config.impl_cfg["use_thd"] is True
    assert mlite_config.impl_cfg["use_deepep"] is False
    assert mlite_config.parallel.tp == 1
    assert mlite_config.parallel.ep == 8
    assert mlite_config.parallel.etp == 1
    assert mlite_config.parallel.pp == 1
    assert mlite_config.parallel.cp == 1
    assert mlite_config.optimizer.offload_fraction is None


def test_welm_mlite_config_maps_deepep_backend(monkeypatch):
    _patch_hf_resolve(monkeypatch)
    config = _finetune_config()
    config.policy.ep_backend = "deepep"

    mlite_config = MliteEngine(config, config.policy, None)._build_mlite_config()

    assert mlite_config.impl_cfg["use_deepep"] is True


def test_welm_mlite_config_rejects_unknown_ep_backend(monkeypatch):
    _patch_hf_resolve(monkeypatch)
    config = _finetune_config()
    config.policy.ep_backend = "unknown"

    engine = MliteEngine(config, config.policy, None)
    with pytest.raises(ValueError, match="mlite ep_backend must be"):
        engine._build_mlite_config()


def test_welm_mlite_config_rejects_hf_model_type_mismatch(monkeypatch):
    _patch_hf_resolve(monkeypatch, model_type="qwen3_5_moe")
    config = _finetune_config()
    engine = MliteEngine(config, config.policy, None)

    with pytest.raises(ValueError, match="requires HF model_type='welmv4_moe'"):
        engine._build_mlite_config()


def test_welm_mlite_config_rejects_pp_gt_one():
    config = _finetune_config()
    config.policy.dist_config.pipeline_model_parallel_size = 2

    with pytest.raises(NotImplementedError, match="pp=1"):
        MliteEngine(config, config.policy, None)


def test_welm_mlite_config_accepts_tp8(monkeypatch):
    _patch_hf_resolve(monkeypatch)
    config = _finetune_config()
    config.policy.dist_config.tensor_model_parallel_size = 8
    config.policy.dist_config.expert_model_parallel_size = 16

    engine = MliteEngine(config, config.policy, None)
    mlite_config = engine._build_mlite_config()

    assert mlite_config.parallel.tp == 8
    assert mlite_config.parallel.ep == 16
    assert mlite_config.parallel.etp == 1


def test_welm_mlite_config_accepts_ep_cp_and_dynamic_cp(monkeypatch):
    _patch_hf_resolve(monkeypatch)
    config = _finetune_config()
    config.policy.dist_config.expert_model_parallel_size = 8
    config.policy.dist_config.context_parallel_size = 1
    config.policy.dist_config.dynamic_context_parallel = True
    config.policy.dist_config.max_seqlen_per_dp_cp_rank = 2048
    config.training.freeze_moe_router = True

    engine = MliteEngine(config, config.policy, None)
    mlite_config = engine._build_mlite_config()

    assert mlite_config.parallel.tp == 1
    assert mlite_config.parallel.ep == 8
    assert mlite_config.parallel.cp == 1
    assert mlite_config.parallel.pp == 1
    assert mlite_config.impl_cfg["cp_attention_backend"] == "all_gather"
    assert "dynamic_context_parallel" in mlite_config.impl_cfg["runtime_plugins"]


def test_welm_mlite_dynamic_cp_metrics_only_count_group_leader(monkeypatch):
    engine = _engine(monkeypatch)
    engine.policy_config.dist_config.dynamic_context_parallel = True
    protocol = SimpleNamespace(
        unpack_forward_output=lambda *_args: torch.tensor([1.0, 2.0])
    )
    engine.handle = SimpleNamespace(
        dp_size=1,
        _model=object(),
        _extras={"protocol": protocol},
    )
    engine._loss_fn = lambda *_args: SimpleNamespace(
        loss=torch.tensor(3.0),
        local_loss_sum=torch.tensor(7.0),
        local_valid_tokens=torch.tensor(2.0),
    )
    runtime_loss_fn = engine._make_runtime_loss_fn(
        global_valid_tokens=torch.tensor(2.0),
        num_microbatches=2,
    )
    loss_context = SimpleNamespace(
        source_batch={
            "aligned_loss_mask": SimpleNamespace(values=lambda: torch.ones(2))
        }
    )

    metrics = {}
    for is_leader in (True, False):
        loss, metrics[is_leader] = runtime_loss_fn(
            {"log_probs": torch.tensor([1.0, 2.0])},
            SimpleNamespace(
                extras={"_mlite_dcp_group_leader": is_leader},
            ),
            loss_context,
        )
        assert torch.equal(loss, torch.tensor(6.0))

    assert torch.equal(metrics[True]["_mlite_loss_sum"], torch.tensor(7.0))
    assert torch.equal(metrics[True]["_mlite_token_count"], torch.tensor(2.0))
    assert torch.equal(metrics[False]["_mlite_loss_sum"], torch.tensor(0.0))
    assert torch.equal(metrics[False]["_mlite_token_count"], torch.tensor(0.0))


@pytest.mark.parametrize(
    "override",
    [
        {"offload_fraction": 1.0},
        {"use_precision_aware_optimizer": True},
        {"decoupled_weight_decay": True},
    ],
)
def test_welm_mlite_config_rejects_unsupported_fsdp2_optimizer_overrides(
    monkeypatch,
    override,
):
    config = _finetune_config()
    config.optimizer.override_optimizer_config = override
    _patch_hf_resolve(monkeypatch)
    engine = MliteEngine(config, config.policy, None)

    with pytest.raises(NotImplementedError, match="keeps optimizer state on GPU"):
        engine._build_mlite_config()


def test_welm_mlite_config_rejects_dcp_until_ep_local_params_are_supported():
    config = _finetune_config()
    config.checkpoint.use_dist_checkpointing = True

    with pytest.raises(NotImplementedError, match="rank-local checkpointing"):
        MliteEngine(config, config.policy, None)


def test_welm_mlite_factory_accepts_finetune(monkeypatch):
    _patch_hf_resolve(monkeypatch)
    config = _finetune_config()

    engine = training_backend.TrainingEngineFactory.get_training_engine(
        config,
        policy_config=config.policy,
        tokenizer=None,
    )

    assert isinstance(engine, MliteEngine)
    assert type(engine.prepare_data).__name__ == "WelmV4PrepareDataForwardLLM"


def test_welm_mlite_engine_validates_dense_dp_layout(monkeypatch):
    engine = _engine(monkeypatch)
    parallel_state = SimpleNamespace(dp_size=32)
    engine.handle = SimpleNamespace(
        dp_size=1,
        _parallel_state=parallel_state,
    )
    monkeypatch.setattr(
        mlite_engine_module.mpu,
        "get_data_parallel_world_size",
        lambda: 32,
    )
    monkeypatch.setattr(
        mlite_engine_module.torch.distributed,
        "is_initialized",
        lambda: False,
    )

    engine._validate_welm_parallel_layout()

    parallel_state.dp_size = 4
    with pytest.raises(RuntimeError, match="dense-DP sizes differ"):
        engine._validate_welm_parallel_layout()


def test_welm_mlite_checkpoint_uses_marker_and_refreshes_optimizer_master(
    monkeypatch,
    tmp_path,
):
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
            self.save_path = None
            self.save_kwargs = None
            self.load_path = None
            self.load_kwargs = None

        def save_checkpoint(self, _handle, path, *, step, **kwargs):
            self.save_path = path
            self.save_kwargs = kwargs
            os.makedirs(path, exist_ok=True)

        def load_checkpoint(self, _handle, path, **kwargs):
            self.load_path = path
            self.load_kwargs = kwargs
            return 7

    class Optimizer:
        def __init__(self):
            self.reload_calls = 0

        def reload_model_params(self):
            self.reload_calls += 1

    scheduler = Scheduler()
    runtime = Runtime()
    optimizer = Optimizer()
    engine.runtime = runtime
    engine.optimizer = optimizer
    engine.handle = SimpleNamespace(_lr_scheduler=scheduler)

    engine.save_checkpoint(7)
    (tmp_path / "step_8").mkdir()

    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "7"
    assert (tmp_path / "step_7" / "lr_scheduler.pt").is_file()
    assert (tmp_path / "step_7" / "rank_local_topology.json").is_file()
    assert runtime.save_path == str(tmp_path / "step_7")
    assert runtime.save_kwargs == {"use_dcp": False}
    assert engine.load_checkpoint(str(tmp_path)) == 7
    assert runtime.load_path == str(tmp_path / "step_7")
    assert runtime.load_kwargs == {"use_dcp": False}
    assert optimizer.reload_calls == 1
    assert scheduler.loaded_state == {"num_steps": 7}


def test_welm_mlite_checkpoint_rejects_topology_mismatch(monkeypatch, tmp_path):
    engine = _engine(monkeypatch)
    engine.checkpoint_config.save_ckpt_path = str(tmp_path)

    class Runtime:
        def save_checkpoint(self, _handle, path, *, step, **kwargs):
            del step, kwargs
            os.makedirs(path, exist_ok=True)

        def load_checkpoint(self, _handle, path, **kwargs):
            del _handle, path, kwargs
            pytest.fail("topology must be validated before loading rank-local state")

    engine.runtime = Runtime()
    engine.optimizer = SimpleNamespace(reload_model_params=lambda: None)
    engine.handle = SimpleNamespace(
        _lr_scheduler=SimpleNamespace(state_dict=lambda: {}),
    )
    engine.save_checkpoint(7)
    engine.policy_config.dist_config.expert_model_parallel_size = 4

    with pytest.raises(RuntimeError, match="checkpoint topology mismatch"):
        engine.load_checkpoint(str(tmp_path))


def test_welm_mlite_checkpoint_nonzero_rank_uses_broadcast_marker(
    monkeypatch,
    tmp_path,
):
    broadcasts = []
    monkeypatch.setattr(mlite_checkpoint_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(mlite_checkpoint_module, "_rank", lambda: 1)
    monkeypatch.setattr(
        mlite_checkpoint_module.os.path,
        "isfile",
        lambda _path: pytest.fail("nonzero rank must not read the marker"),
    )

    def broadcast_marker(payload, src):
        assert src == 0
        payload[:] = [7, None]
        broadcasts.append(tuple(payload))

    monkeypatch.setattr(
        mlite_checkpoint_module.dist,
        "broadcast_object_list",
        broadcast_marker,
    )

    step, step_path = mlite_checkpoint_module._resolve_rank_local_resume_path(
        str(tmp_path)
    )

    assert step == 7
    assert step_path == str(tmp_path / "step_7")
    assert broadcasts == [(7, None)]


def test_welm_mlite_recipe_uses_validated_training_topology():
    recipe_path = (
        Path(__file__).resolve().parents[2]
        / "tasks"
        / "welm_v4_5"
        / "yaml"
        / "sft_mlite.yaml"
    )
    recipe = OmegaConf.load(recipe_path)

    assert recipe.training.training_backend == "mlite"
    assert recipe.training.freeze_moe_router is False
    assert recipe.training.manual_gc is True
    assert recipe.training.manual_gc_interval == 20
    assert recipe.policy.model_arch == "welmv4_moe"
    assert recipe.policy.dist_config.tensor_model_parallel_size == 1
    assert recipe.policy.dist_config.expert_model_parallel_size == 16
    assert recipe.policy.dist_config.expert_tensor_parallel_size == 1
    assert recipe.policy.dist_config.context_parallel_size == 1
    assert recipe.policy.dist_config.dynamic_context_parallel is True
    assert recipe.policy.dist_config.max_seqlen_per_dp_cp_rank == 8192
    assert recipe.policy.dist_config.min_dynamic_context_parallel_size == 1
    assert recipe.policy.wrap_with_ddp is False
    assert recipe.checkpoint.use_dist_checkpointing is False
