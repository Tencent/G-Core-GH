# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
# nrwu@tencent.com

"""HpModule marker class.

Kept in its own module to avoid importing the parent package's heavy
``__init__.py`` (diffusers, etc.) and to prevent circular imports.
"""


class HpModule:
    """Marker base class for hybrid-parallel models.

    Subclasses opt into the alternative construction path in
    ``Fsdp2EngineMixin``: meta-device init -> ``apply_hp`` -> binds
    ``clip_grad_norm_`` and ``load_checkpoint_hp`` /
    ``load_state_dict_hp``, instead of the default ``from_pretrained``
    + FSDP2 wrap flow.
    """
    pass
