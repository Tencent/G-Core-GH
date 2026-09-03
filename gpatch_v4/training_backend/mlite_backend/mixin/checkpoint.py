from gpatch_v4.training_backend.mlite_backend.checkpoint import (
    load_checkpoint as load_mlite_checkpoint,
)
from gpatch_v4.training_backend.mlite_backend.checkpoint import (
    save_checkpoint as save_mlite_checkpoint,
)


class CheckpointMixin:
    def save_checkpoint(self, global_step: int, dataloader=None) -> None:
        self._require_initialized()
        save_mlite_checkpoint(self, global_step, dataloader=dataloader)

    def load_checkpoint(self, path: str) -> int:
        self._require_initialized()
        return load_mlite_checkpoint(self, path)
