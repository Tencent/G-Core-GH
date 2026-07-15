import numpy as np
import torch

try:
    import pybase64
except ImportError:
    pybase64 = None

try:
    # sglang moved this out of layers.moe (e.g. dsv4 / recent main)
    from sglang.srt.state_capturer.routed_experts import (
        extract_routed_experts_from_meta_info,
    )
except ImportError:
    try:
        from sglang.srt.layers.moe.routed_experts_capturer import (
            extract_routed_experts_from_meta_info,
        )
    except ImportError:
        extract_routed_experts_from_meta_info = None


def extract_routed_experts(res):
    """Extract routed expert info from an sglang response meta_info.

    Parameters
    ----------
    res : dict
        Generation response with ``'meta_info'``.

    Returns
    -------
    object or None
        Routed experts data, or *None* if unavailable.
    """
    routed_experts = res["meta_info"].get("routed_experts", None)
    return routed_experts


def process_routed_experts(res, num_layers, moe_router_topk, return_dtype=torch.int32):
    """Parse and reshape routed expert indices from a generation result.

    Parameters
    ----------
    res : SimpleNamespace
        Generation result with ``routed_experts``, ``prompt_len``, and ``token_ids``.
    num_layers : int
    moe_router_topk : int
    return_dtype : torch.dtype, optional

    Returns
    -------
    torch.Tensor or None
        Routed expert tensor of shape ``(seq_len, num_layers, topk)``.
    """
    routed_experts = res.routed_experts
    if routed_experts is None:
        return None

    # backward compatibility, process list of list of list [token, layer, moe_topk]
    if isinstance(routed_experts, list):
        assert len(routed_experts) == res.prompt_len + len(res.token_ids), \
            f"{len(routed_experts)} != {res.prompt_len} + {len(res.token_ids)}"
        # TODO(hessianliu): handle first replace dense
        # # seq, layer, experts
        # for (ti, t) in enumerate(routed_experts):
        #     for (li, layer) in enumerate(t):
        #         assert len(layer) == len(set(layer)), f"{ti}  {li} {layer}, {output.outputs[0].prompt_len} {len(output.outputs[0].token_ids)}"
        routed_experts = torch.tensor(routed_experts, dtype=return_dtype)
        return routed_experts

    if isinstance(routed_experts, dict):
        assert extract_routed_experts_from_meta_info is not None
        routed_experts = extract_routed_experts_from_meta_info(
            {"meta_info": {
                "routed_experts": routed_experts
            }}
        )

    if isinstance(routed_experts, np.ndarray):
        routed_experts = torch.from_numpy(routed_experts)
    elif isinstance(routed_experts, torch.Tensor):
        pass
    else:
        # sglang >=0.5.7 base64 encoded string
        assert extract_routed_experts_from_meta_info is not None and pybase64 is not None, (
            "routed_experts is base64-encoded but sglang extract helper / pybase64 "
            "is unavailable; check sglang install and PYTHONPATH"
        )
        routed_experts = np.frombuffer(
            pybase64.b64decode(routed_experts.encode("utf-8")), dtype=np.int32
        ).reshape(-1, num_layers, moe_router_topk)
        routed_experts = torch.from_numpy(routed_experts)
    assert (routed_experts.shape[0] + 1) == res.prompt_len + len(res.token_ids), \
        f"{routed_experts.shape[0] + 1} != {res.prompt_len} + {len(res.token_ids)}"
    routed_experts = routed_experts.to(return_dtype)
    # backward compatibility, append one token at the end
    # TODO(hessianliu): rm this for v4
    routed_experts = torch.cat([routed_experts, routed_experts[-1:]])
    return routed_experts
