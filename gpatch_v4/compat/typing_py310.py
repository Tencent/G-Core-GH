import sys


def ensure_typing_self() -> None:
    """Patch ``typing`` with typing_extensions names missing on Python <3.12.

    Do not backport TypeAliasType: typing_extensions treats its presence on
    typing as a 3.12 signal and then calls ParamSpec(infer_variance=...),
    which 3.10 rejects and breaks import torch.
    """
    if sys.version_info >= (3, 11):
        return

    import typing

    import typing_extensions as te

    for name in (
        "Self",  # 3.11; Megatron-Bridge
        "Unpack",
        "NotRequired",
        "Required",
        "TypeIs",
        "assert_never",
        "override",  # 3.12; Megatron-LM gpt.py
    ):
        if not hasattr(typing, name) and hasattr(te, name):
            setattr(typing, name, getattr(te, name))
