import os


def sglang_hack():
    for k in ["TORCHELASTIC_USE_AGENT_STORE"]:
        if k in os.environ:
            del os.environ[k]
    os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"

    try:
        from sglang.srt.patch_torch import monkey_patch_torch_reductions
    except:
        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
    monkey_patch_torch_reductions()
