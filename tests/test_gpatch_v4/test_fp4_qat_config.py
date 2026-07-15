from types import SimpleNamespace

from gpatch_v4.configs.policy_config import BasePolicyConfig
from gpatch_v4.models.hp_module import HpModule
from gpatch_v4.training_backend.fsdp2_backend.mixin import Fsdp2EngineMixin
import gpatch_v4.training_backend.fsdp2_backend.mixin as fsdp2_mixin


class _FakeConfig:
    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return SimpleNamespace(
            num_nextn_predict_layers=0,
            num_hidden_layers=1,
            layer_types=None,
            mlp_layer_types=None,
        )


class _FakeHpModel(HpModule):
    config_class = _FakeConfig

    def __init__(self, config):
        self.config = config

    def load_checkpoint_hp(self, _path):
        pass

    def train(self):
        return self

    def eval(self):
        return self


def test_fsdp2_mixin_passes_fp4_qat_to_apply_hp(monkeypatch):
    captured = {}

    def fake_apply_hp(model, mesh, **kwargs):
        captured["model"] = model
        captured["mesh"] = mesh
        captured.update(kwargs)
        return model

    monkeypatch.setattr(fsdp2_mixin, "apply_hp", fake_apply_hp)
    engine = SimpleNamespace(
        get_model_cls=lambda: _FakeHpModel,
        training_config=SimpleNamespace(enable_mtp=False),
        policy_config=BasePolicyConfig(fp4_qat=True, fp8_qat=True),
        config=SimpleNamespace(
            debug=SimpleNamespace(debug_truncate_num_hidden_layers=None),
        ),
        ep_2d_mesh=object(),
        cp_mesh_for_hp=None,
    )

    model = Fsdp2EngineMixin.get_fsdp2_model(engine, None, "checkpoint")

    assert model is captured["model"]
    assert captured["fp4_qat"]
    assert captured["fp8_qat"]
