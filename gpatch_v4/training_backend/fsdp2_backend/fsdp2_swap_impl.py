import torch

from gpatch_v4.training_backend.fsdp2_backend.swap import (
    offload_model,
    offload_optimizer,
    onload_model,
    onload_optimizer,
)
from gpatch_v4.utils.common_utils import clear_memory


class Fsdp2SwapImpl:
    @classmethod
    @torch.no_grad()
    def release_grad(cls, models):
        """Release gradient data to free GPU memory without offloading model params."""
        if models is None:
            return
        if isinstance(models, torch.nn.Module):
            models = [models]
        for model in models:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = None
        clear_memory()

    @classmethod
    @torch.no_grad()
    def offload_model(cls, models):
        offload_model(models)

    @classmethod
    @torch.no_grad()
    def onload_model(cls, models, onload_grad=True):
        onload_model(models)

    @classmethod
    @torch.no_grad()
    def offload_optimizer(cls, optimizers):
        clear_memory()
        offload_optimizer(optimizers)

    @classmethod
    @torch.no_grad()
    def onload_optimizer(cls, optimizers):
        clear_memory()
        onload_optimizer(optimizers)
