from types import SimpleNamespace

from gpatch_v4.models.deepseek_v4.mtp import config_for_synthetic_layer


def _parent_config():
    return SimpleNamespace(
        layer_types=["sliding_attention", "sliding_attention"],
        mlp_layer_types=["moe", "moe"],
        compress_ratios=[0, 0],
        attn_backend="eager",
        fp8_qat=False,
    )


def test_synthetic_layer_config_reuses_parent_when_in_range():
    parent = _parent_config()
    assert config_for_synthetic_layer(parent, layer_idx=1) is parent


def test_synthetic_layer_config_pads_lists_and_shares_runtime_knobs():
    parent = _parent_config()
    cfg = config_for_synthetic_layer(parent, layer_idx=3)

    assert cfg is not parent
    assert cfg.layer_types == [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
    ]
    assert cfg.mlp_layer_types == ["moe", "moe", "moe", "moe"]
    assert cfg.compress_ratios is parent.compress_ratios
    assert parent.layer_types == ["sliding_attention", "sliding_attention"]

    parent.attn_backend = "fused"
    assert cfg.attn_backend == "fused"

    cfg.fp8_qat = True
    assert parent.fp8_qat is True
    assert cfg.fp8_qat is True


def test_synthetic_layer_config_ignores_compress_ratios_length():
    parent = _parent_config()
    parent.compress_ratios = [0]
    assert config_for_synthetic_layer(parent, layer_idx=1) is parent
