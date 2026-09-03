import os
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MEGATRON_FSDP_INTEGRATION", "0") != "1",
    reason="default skipped; set RUN_MEGATRON_FSDP_INTEGRATION=1 to enable",
)


def test_policy_config_megatron_fsdp_defaults():
    from gpatch_v4.configs.policy_config import BasePolicyConfig

    config = BasePolicyConfig()

    assert config.use_megatron_fsdp is False
    assert config.override_ddp_config == {}


def test_megatron_fsdp_ddp_config_defaults_and_overrides():
    mixin = pytest.importorskip("gpatch_v4.training_backend.megatron_backend.mixin")

    ddp_config = mixin.build_megatron_bridge_ddp_config_dict(
        use_megatron_fsdp=True,
        override_ddp_config={"overlap_grad_reduce": False},
        build_from_mbridge=False,
        wrap_with_ddp=True,
    )

    assert ddp_config["use_distributed_optimizer"] is True
    assert ddp_config["use_megatron_fsdp"] is True
    assert ddp_config["data_parallel_sharding_strategy"] == "optim_grads_params"
    assert ddp_config["check_for_nan_in_grad"] is True
    assert ddp_config["overlap_grad_reduce"] is False


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"build_from_mbridge": True}, "build_from_mbridge=False"),
        ({"wrap_with_ddp": False}, "wrap_with_ddp=True"),
        ({"override_ddp_config": {"use_distributed_optimizer": False}}, "use_distributed_optimizer"),
        ({"override_ddp_config": {"use_megatron_fsdp": False}}, "use_megatron_fsdp"),
        ({"override_ddp_config": {"data_parallel_sharding_strategy": "optim_grads"}}, "data_parallel"),
    ],
)
def test_megatron_fsdp_ddp_config_rejects_invalid_invariants(kwargs, match):
    mixin = pytest.importorskip("gpatch_v4.training_backend.megatron_backend.mixin")
    params = {
        "use_megatron_fsdp": True,
        "override_ddp_config": {},
        "build_from_mbridge": False,
        "wrap_with_ddp": True,
    }
    params.update(kwargs)

    with pytest.raises(AssertionError, match=match):
        mixin.build_megatron_bridge_ddp_config_dict(**params)


def test_megatron_fsdp_provider_kwargs_disable_init_broadcast():
    mixin = pytest.importorskip("gpatch_v4.training_backend.megatron_backend.mixin")
    ddp_config = object()

    kwargs = mixin.build_megatron_bridge_provide_model_kwargs(
        wrap_with_ddp=True,
        ddp_config=ddp_config,
        use_megatron_fsdp=True,
    )

    assert kwargs["wrap_with_ddp"] is True
    assert kwargs["ddp_config"] is ddp_config
    assert kwargs["use_megatron_fsdp"] is True
    assert kwargs["data_parallel_random_init"] is False


def test_megatron_fsdp_weight_load_keeps_wrapped_model(monkeypatch):
    mixin = pytest.importorskip("gpatch_v4.training_backend.megatron_backend.mixin")
    wrapped_model = object()
    unwrap_mock = Mock()
    monkeypatch.setattr(mixin, "unwrap_model", unwrap_mock)

    result = mixin.get_megatron_bridge_weight_load_model(
        wrapped_model,
        use_megatron_fsdp=True,
    )

    assert result is wrapped_model
    unwrap_mock.assert_not_called()


def test_non_fsdp_weight_load_keeps_wrapped_model(monkeypatch):
    mixin = pytest.importorskip("gpatch_v4.training_backend.megatron_backend.mixin")
    wrapped_model = object()
    unwrap_mock = Mock()
    monkeypatch.setattr(mixin, "unwrap_model", unwrap_mock)

    result = mixin.get_megatron_bridge_weight_load_model(
        wrapped_model,
        use_megatron_fsdp=False,
    )

    assert result is wrapped_model
    unwrap_mock.assert_not_called()


def _fsdp_config():
    return SimpleNamespace(
        training=SimpleNamespace(build_from_mbridge=False),
        checkpoint=SimpleNamespace(
            save_ckpt_path="save",
            load_ckpt_path="load",
            async_save=False,
            skip_save_mcore_model=False,
            convert_mcore_to_hf_online=False,
            no_save_optim=False,
            no_load_optim=False,
        ),
    )


def test_megatron_fsdp_checkpoint_scope_assertions():
    checkpoint = pytest.importorskip("gpatch_v4.training_backend.megatron_backend.checkpoint")
    config = _fsdp_config()

    checkpoint._assert_megatron_fsdp_checkpoint_supported(config, [object()], is_save=True)

    config.checkpoint.async_save = True
    with pytest.raises(AssertionError, match="async_save"):
        checkpoint._assert_megatron_fsdp_checkpoint_supported(config, [object()], is_save=True)


def test_megatron_fsdp_save_uses_bridge_preprocess_and_torch_dist_save(monkeypatch, tmp_path):
    checkpoint = pytest.importorskip("gpatch_v4.training_backend.megatron_backend.checkpoint")
    bridge_checkpointing = pytest.importorskip("megatron.bridge.training.checkpointing")
    torch_dist_checkpoint = pytest.importorskip("torch.distributed.checkpoint")

    raw_optimizer = {"state": {}, "param_to_group_meta": {}}
    raw_state_dict = {"model": {"weight": object()}, "optimizer": raw_optimizer, "rng_state": {}}
    processed_state_dict = {"processed": True}
    monkeypatch.setattr(checkpoint, "generate_state_dict", Mock(return_value=raw_state_dict))
    monkeypatch.setattr(checkpoint, "get_model_config", Mock(return_value=SimpleNamespace()))
    normalize_mock = Mock(side_effect=lambda model, opt: opt)
    monkeypatch.setattr(checkpoint, "normalize_megatron_fsdp_optimizer_state_tensors", normalize_mock)
    monkeypatch.setattr(
        bridge_checkpointing,
        "preprocess_fsdp_dtensor_state_dict",
        Mock(return_value=processed_state_dict),
    )
    monkeypatch.setattr(torch_dist_checkpoint, "FileSystemWriter", Mock(return_value="writer"))
    save_mock = Mock()
    monkeypatch.setattr(torch_dist_checkpoint, "save", save_mock)
    monkeypatch.setattr(checkpoint.torch.distributed, "barrier", Mock())

    model = SimpleNamespace(state_dict_for_save_checkpoint=Mock(), module=SimpleNamespace())
    checkpoint.save_megatron_fsdp_checkpoint(
        _fsdp_config(),
        [model],
        optimizer=object(),
        lr_scheduler=object(),
        dist_checkpoint_path=str(tmp_path / "iter_0000001"),
    )

    normalize_mock.assert_called_once_with(model.module, raw_optimizer)
    bridge_checkpointing.preprocess_fsdp_dtensor_state_dict.assert_called_once()
    save_mock.assert_called_once_with(state_dict=processed_state_dict, storage_writer="writer")


def test_megatron_fsdp_load_uses_local_bridge_signature(monkeypatch):
    checkpoint = pytest.importorskip("gpatch_v4.training_backend.megatron_backend.checkpoint")
    bridge_checkpointing = pytest.importorskip("megatron.bridge.training.checkpointing")

    model = SimpleNamespace(
        module=SimpleNamespace(),
        load_state_dict=Mock(),
    )
    optimizer = SimpleNamespace(load_state_dict=Mock())
    lr_scheduler = SimpleNamespace(load_state_dict=Mock())
    state_dict = {
        "model": {"weight": object()},
        "optimizer": {"state": object()},
        "lr_scheduler": {"lr": object()},
        "rng_state": {"(0, 0)": [{"rng": object()}]},
    }

    monkeypatch.setattr(checkpoint, "generate_state_dict", Mock(return_value={"model": {}}))
    monkeypatch.setattr(checkpoint, "get_model_config", Mock(return_value=SimpleNamespace()))
    # create_autospec keeps real Bridge kwargs so production inspect.signature check passes.
    load_mock = create_autospec(
        bridge_checkpointing.load_fsdp_dtensor_checkpoint,
        return_value=(state_dict, "checkpoint", False, object()),
    )
    monkeypatch.setattr(bridge_checkpointing, "load_fsdp_dtensor_checkpoint", load_mock)
    monkeypatch.setattr(checkpoint, "load_rng_states", Mock())

    checkpoint.load_megatron_fsdp_checkpoint(
        _fsdp_config(),
        [model],
        optimizer,
        lr_scheduler,
        global_step=3,
    )

    load_mock.assert_called_once()
    call_kwargs = load_mock.call_args.kwargs
    assert call_kwargs["load_dir"] == "load"
    assert call_kwargs["iteration"] == 3
    assert call_kwargs["ckpt_cfg"].ckpt_format == "fsdp_dtensor"
    model.load_state_dict.assert_called_once_with(state_dict["model"], strict=True)
    optimizer.load_state_dict.assert_called_once_with(state_dict["optimizer"])
    lr_scheduler.load_state_dict.assert_called_once_with(state_dict["lr_scheduler"])
    checkpoint.load_rng_states.assert_called_once_with(state_dict["rng_state"], use_megatron_fsdp=True)


def _make_fake_fsdp_dist_param(*, global_numel: int, fsdp_slice: slice, local_shape):
    torch = pytest.importorskip("torch")

    full = torch.arange(global_numel, dtype=torch.float32)
    param = torch.nn.Parameter(full.clone())
    param.megatron_fsdp_slice = fsdp_slice
    param.megatron_fsdp_dist_index = object()
    param._local_tensor = full[fsdp_slice].reshape(local_shape).contiguous()
    return param


def test_plain_tensor_to_fsdp_local_shard_slices_global_moment():
    torch = pytest.importorskip("torch")
    checkpoint = pytest.importorskip("gpatch_v4.training_backend.megatron_backend.checkpoint")

    global_numel = 20
    fsdp_slice = slice(8, 14)
    dist_param = _make_fake_fsdp_dist_param(
        global_numel=global_numel,
        fsdp_slice=fsdp_slice,
        local_shape=(2, 3),
    )
    full = torch.arange(global_numel, dtype=torch.float32)

    local = checkpoint.plain_tensor_to_fsdp_local_shard(full, dist_param)

    assert local.shape == (2, 3)
    assert torch.equal(local.view(-1), full.view(-1)[fsdp_slice])


def test_plain_tensor_to_fsdp_local_shard_accepts_already_local():
    torch = pytest.importorskip("torch")
    checkpoint = pytest.importorskip("gpatch_v4.training_backend.megatron_backend.checkpoint")

    fsdp_slice = slice(8, 14)
    dist_param = _make_fake_fsdp_dist_param(
        global_numel=20,
        fsdp_slice=fsdp_slice,
        local_shape=(2, 3),
    )
    already_local = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    local = checkpoint.plain_tensor_to_fsdp_local_shard(already_local, dist_param)

    assert torch.equal(local, already_local)


def test_plain_tensor_to_fsdp_local_shard_rejects_bad_numel():
    torch = pytest.importorskip("torch")
    checkpoint = pytest.importorskip("gpatch_v4.training_backend.megatron_backend.checkpoint")

    dist_param = _make_fake_fsdp_dist_param(
        global_numel=20,
        fsdp_slice=slice(8, 14),
        local_shape=(2, 3),
    )

    with pytest.raises(ValueError, match="does not match FSDP global or local shard"):
        checkpoint.plain_tensor_to_fsdp_local_shard(torch.arange(7), dist_param)


def test_normalize_megatron_fsdp_optimizer_state_tensors_wraps_plain(monkeypatch):
    torch = pytest.importorskip("torch")
    checkpoint = pytest.importorskip("gpatch_v4.training_backend.megatron_backend.checkpoint")

    fsdp_slice = slice(4, 10)
    dist_param = _make_fake_fsdp_dist_param(
        global_numel=16,
        fsdp_slice=fsdp_slice,
        local_shape=(2, 3),
    )
    # Nested modules so named path can contain '.' (register_parameter forbids '.').
    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.decoder.layers[0].mlp = torch.nn.Module()
    model.decoder.layers[0].mlp.linear_fc1 = torch.nn.Module()
    model.decoder.layers[0].mlp.linear_fc1.register_parameter("weight", dist_param)

    full = torch.arange(16, dtype=torch.float32)
    optimizer_state_dict = {
        "state": {
            "decoder.layers.0.mlp.linear_fc1.weight": {
                "exp_avg": full.clone(),
                "exp_avg_sq": full.clone() + 100,
                "step": torch.tensor(3),
            }
        },
        "param_to_group_meta": {},
    }

    wrapped = []

    def fake_make_fsdp_dtensor(local_tensor, param, dist_index, **kwargs):
        wrapped.append((local_tensor.clone(), param, dist_index, kwargs))
        return ("dtensor", local_tensor.numel())

    monkeypatch.setattr(
        "megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer.make_fsdp_dtensor",
        fake_make_fsdp_dtensor,
    )

    out = checkpoint.normalize_megatron_fsdp_optimizer_state_tensors(model, optimizer_state_dict)

    st = out["state"]["decoder.layers.0.mlp.linear_fc1.weight"]
    assert st["exp_avg"] == ("dtensor", 6)
    assert st["exp_avg_sq"] == ("dtensor", 6)
    assert torch.equal(st["step"], torch.tensor(3))
    assert len(wrapped) == 2
    assert torch.equal(wrapped[0][0].view(-1), full.view(-1)[fsdp_slice])
    assert wrapped[0][1] is dist_param
