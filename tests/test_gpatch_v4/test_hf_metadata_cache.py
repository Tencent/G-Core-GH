import shutil
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gpatch_v4.utils import common_utils


def test_hf_metadata_cache_survives_origin_removal(tmp_path, monkeypatch):
    monkeypatch.setattr(common_utils, "HF_METADATA_CACHE_ROOT", str(tmp_path / "cache"))

    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    (hf_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (hf_dir / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    (hf_dir / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    (hf_dir / "model.safetensors").write_text("weights", encoding="utf-8")

    save_path = tmp_path / "save"

    common_utils.cache_hf_metadata_files(hf_dir, save_path)
    shutil.rmtree(hf_dir)

    export_dir = tmp_path / "export"
    common_utils.copy_cached_hf_metadata_files(hf_dir, save_path, export_dir)

    assert (export_dir / "config.json").exists()
    assert (export_dir / "tokenizer_config.json").exists()
    assert (export_dir / "chat_template.jinja").exists()
    assert not (export_dir / "model.safetensors.index.json").exists()
    assert not (export_dir / "model.safetensors").exists()


def test_assert_hf_metadata_cache_exists_requires_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(common_utils, "HF_METADATA_CACHE_ROOT", str(tmp_path / "cache"))

    with pytest.raises(AssertionError, match="HF metadata cache .* does not exist"):
        common_utils.assert_hf_metadata_cache_exists(
            tmp_path / "missing_hf",
            tmp_path / "save",
        )


def test_mbridge_save_hf_uses_existing_safetensor_io(tmp_path, monkeypatch):
    checkpoint = pytest.importorskip("gpatch_v4.training_backend.megatron_backend.checkpoint")
    monkeypatch.setattr(common_utils, "HF_METADATA_CACHE_ROOT", str(tmp_path / "cache"))

    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    save_path = tmp_path / "save"
    common_utils.cache_hf_metadata_files(hf_dir, save_path)
    shutil.rmtree(hf_dir)

    config = SimpleNamespace(
        checkpoint=SimpleNamespace(
            export_hf_save_path=None,
            save_ckpt_path=str(save_path),
            mbridge_distributed_filesystem=False,
            strict_export=True,
        ),
        policy=SimpleNamespace(hf_model_path=str(hf_dir)),
    )
    bridge = SimpleNamespace(
        safetensor_io=object(),
        _get_safetensor_io=Mock(side_effect=AssertionError("_get_safetensor_io should not run")),
        save_weights=Mock(),
    )

    monkeypatch.setattr(checkpoint, "cpu_barrier", Mock())
    monkeypatch.setattr(checkpoint, "sync_cuda_and_get_time", Mock(side_effect=[1.0, 2.0]))
    monkeypatch.setattr(checkpoint, "unwrap_model", Mock(return_value=["model"]))
    monkeypatch.setattr(checkpoint, "save_args_json", Mock())
    monkeypatch.setattr(checkpoint.torch.distributed, "get_rank", Mock(return_value=0))

    checkpoint._mbridge_save_hf(config, 3, ["wrapped"], bridge)

    bridge._get_safetensor_io.assert_not_called()
    bridge.save_weights.assert_called_once()
