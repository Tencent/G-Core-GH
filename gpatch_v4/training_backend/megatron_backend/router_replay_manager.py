from typing import List, Optional

import torch

try:
    from megatron.core.transformer.moe.router_replay import (
        RouterReplay,
        RouterReplayAction,
    )
except Exception:
    RouterReplay = None
    RouterReplayAction = None


class RouterReplayManager:

    ROUTER_REPLAY_MGR = None

    @classmethod
    def init_instance(cls):
        assert cls.ROUTER_REPLAY_MGR is None
        cls.ROUTER_REPLAY_MGR = RouterReplayManager()

    @classmethod
    def get_instance(cls):
        if cls.ROUTER_REPLAY_MGR is None:
            cls.init_instance()
        return cls.ROUTER_REPLAY_MGR

    def __init__(self):
        self._enabled = False
        self._model_instances: Optional[List[RouterReplay]] = None

    def __enter__(self):
        self._enabled = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._enabled = False
        self.clear_indices()

    @property
    def enabled(self):
        return self._enabled

    @property
    def model_instances(self) -> Optional[List]:
        return self._model_instances

    def set_model_instances(self, model):
        """Collect RouterReplay instances that belong to the given model.

        Traverses model modules to find TopKRouters with router_replay
        attribute, ensuring we only use this model's instances regardless
        of what is in the global list.
        """
        instances = []
        for module in model.modules():
            replay = getattr(module, "router_replay", None)
            if isinstance(replay, RouterReplay):
                instances.append(replay)
        self._model_instances = instances if instances else None

    def _iter_instances(self):
        if self._model_instances is not None:
            return self._model_instances
        return RouterReplay.global_router_replay_instances

    def set_replay_action(self, action: "RouterReplayAction"):
        for inst in self._iter_instances():
            inst.set_router_replay_action(action)

    def clear_replay_action(self):
        for inst in self._iter_instances():
            inst.clear_router_replay_action()

    def clear_indices(self):
        for inst in self._iter_instances():
            inst.clear_indices()

    def append_micro_batch(self, microbatch):
        """Distribute replay tensors to the model's RouterReplay instances."""
        assert RouterReplay is not None
        instances = list(self._iter_instances())
        if len(microbatch) != len(instances):
            raise ValueError(
                f"The number of replay tensors ({len(microbatch)}) "
                f"does not match model instances ({len(instances)})."
            )
        for i, inst in enumerate(instances):
            inst.set_target_indices(microbatch[i])


RouterReplayCtx = RouterReplayManager
