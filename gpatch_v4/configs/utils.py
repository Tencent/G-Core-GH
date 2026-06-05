import collections
import dataclasses
from dataclasses import asdict
from typing import Type, TypeVar

from omegaconf import MISSING, DictConfig, OmegaConf, open_dict

_T = TypeVar("_T")


# use mapping-protol to make dataclass behaving like a dict for backward compatibility
class MappingProtocol(collections.abc.Mapping):
    """Mixin that makes dataclasses behave like read-only dicts.

    Used for backward compatibility so that config dataclasses can be
    accessed via ``config['key']`` syntax.
    """
    def __getitem__(self, key):
        return asdict(self)[key]

    def __iter__(self):
        return iter(asdict(self))

    def __len__(self):
        return len(asdict(self))


def _create_defaults_instance(cls):
    """Create a dataclass instance with default values, bypassing
    ``__init__`` and ``__post_init__``.

    For fields whose ``default_factory`` is itself a dataclass class,
    the function recurses so that the entire config tree is built
    without triggering any validation.  Non-dataclass factories
    (``list``, ``dict``, lambdas, …) are simply called.
    """
    instance = object.__new__(cls)
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING:
            setattr(instance, f.name, f.default)
        elif f.default_factory is not dataclasses.MISSING:
            factory = f.default_factory
            if dataclasses.is_dataclass(factory):
                setattr(instance, f.name, _create_defaults_instance(factory))
            else:
                setattr(instance, f.name, factory())
        else:
            setattr(instance, f.name, MISSING)
    return instance


def merge_hydra_config(config_cls: Type[_T], cfg: DictConfig) -> _T:
    """Merge a Hydra ``DictConfig`` with dataclass defaults.

    Unlike ``OmegaConf.merge(ConfigClass(), cfg)`` followed by ``to_object``,
    this helper ensures ``__post_init__`` is invoked **exactly once** — on
    the final merged values.

    Parameters
    ----------
    config_cls : type
        Top-level dataclass config class (e.g. ``RlConfig``).
    cfg : DictConfig
        Hydra-resolved configuration.

    Returns
    -------
    _T
        Fully-constructed dataclass instance of ``config_cls``.
    """
    if "_target_" in cfg:
        with open_dict(cfg):
            del cfg["_target_"]
    defaults = _create_defaults_instance(config_cls)
    schema = OmegaConf.structured(defaults)
    merged = OmegaConf.merge(schema, cfg)
    return OmegaConf.to_object(merged)
