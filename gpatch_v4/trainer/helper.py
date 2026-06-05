import os
import re
import time
from dataclasses import fields, is_dataclass

from gpatch_v4.configs.dist_config import DistConfig
from gpatch_v4.configs.utils import MappingProtocol
from gpatch_v4.orches.placement_group import create_train_group


async def convert_mcore_to_hf(config, pgs):
    """Convert a Megatron-Core checkpoint to HuggingFace format.

    Forces loading from ``save_ckpt_path`` and delegates the actual
    conversion to the train group.

    Parameters
    ----------
    config : object
    pgs : dict
    """
    # 转 ckpt 时候，强制让从 save_ckpt_path 加载
    config.training.auto_load_from_save_ckpt = True
    config.checkpoint.load_ckpt_path = config.checkpoint.save_ckpt_path

    train_group = create_train_group(config, pgs)
    await train_group.init()
    await train_group.convert_to_hf_checkpoint()
    return


def get_nnodes() -> int:
    """Return the number of Ray nodes in the cluster.

    Validates the node count against ``/etc/mpi/hostfile`` (if present)
    and the available GPU resources.

    Returns
    -------
    int
    """
    # TODO(@guanyouhe): trainer 避免直接依赖 ray，应该依赖 orches。
    # TODO(@guanyouhe): trainer 避免直接依赖 ray，应该依赖 orches。
    # TODO(@guanyouhe): trainer 避免直接依赖 ray，应该依赖 orches。
    import ray

    def is_node_ipv4_format(text: str) -> bool:
        """
        验证字符串是否为 "node:ipv4" 格式
        """
        pattern = r'^node:(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
        return bool(re.match(pattern, text))

    nnodes = len([n for n in ray.nodes() if n.get("Alive", False)])

    if os.path.isfile("/root/hostfile"):
        with open("/root/hostfile") as f:
            lines = f.readlines()
            expected_nnodes = len(lines)
            # After ray.shutdown()/ray.init(), worker nodes may need time to
            # reconnect to the GCS.  Wait up to 60 seconds for all nodes.
            if nnodes < expected_nnodes:
                for _ in range(60):
                    time.sleep(1)
                    nnodes = len([n for n in ray.nodes() if n.get("Alive", False)])
                    if nnodes >= expected_nnodes:
                        break
            assert expected_nnodes == nnodes, f"hostfile has {expected_nnodes} lines, expected {nnodes}"

    cluster_res = ray.cluster_resources()
    assert int(cluster_res["GPU"]) == nnodes * 8
    nodes_key = []
    for k in cluster_res.keys():
        if isinstance(k, str) and is_node_ipv4_format(k):
            nodes_key.append(k)
    assert nnodes == len(nodes_key)

    return nnodes


def set_nnodes_default(cfg, nnodes=None):
    """Recursively set ``DistConfig.nnodes`` to the detected node count.

    Walks every dataclass field of *cfg* and sets ``nnodes`` on
    ``DistConfig`` instances whose value is ``-1`` (sentinel).

    Parameters
    ----------
    cfg : dataclass
    nnodes : int, optional
        Auto-detected via ``get_nnodes`` when *None*.
    """
    if not is_dataclass(cfg):
        return
    if nnodes is None:
        nnodes = get_nnodes()

    for field in fields(cfg):
        field_name = field.name
        field_value = getattr(cfg, field_name)
        if isinstance(field_value, MappingProtocol):
            if isinstance(field_value, DistConfig):
                if field_value.nnodes == -1:
                    field_value.nnodes = nnodes
            else:
                set_nnodes_default(field_value, nnodes)
        elif isinstance(field_value, list):
            for field_ele in field_value:
                set_nnodes_default(field_ele, nnodes)
        elif isinstance(field_value, dict):
            _resolve_nnodes_in_dict(field_value, nnodes)


def _resolve_nnodes_in_dict(d, nnodes):
    """Recursively resolve ``nnodes: -1`` inside plain dicts (e.g. teachers)."""
    for v in d.values():
        if isinstance(v, dict):
            if 'nnodes' in v and v['nnodes'] == -1:
                v['nnodes'] = nnodes
            _resolve_nnodes_in_dict(v, nnodes)
        elif is_dataclass(v):
            set_nnodes_default(v, nnodes)
