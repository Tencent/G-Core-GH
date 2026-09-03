import torch


class MliteSwapImpl:
    def __init__(self):
        self.runtime = None
        self.handle = None

    def bind(self, runtime, handle) -> None:
        self.runtime = runtime
        self.handle = handle

    def _require_bound(self) -> None:
        if self.runtime is None or self.handle is None:
            raise RuntimeError("MliteSwapImpl is not bound; call bind(runtime, handle) after setup")

    @torch.no_grad()
    def release_grad(self, models) -> None:
        del models
        self._require_bound()
        model = self.handle._model
        chunks = list(model) if isinstance(model, (list, tuple)) else [model]
        for chunk in chunks:
            for parameter in chunk.parameters():
                parameter.grad = None

    @torch.no_grad()
    def offload_model(self, models, tag="") -> None:
        del models, tag
        self._require_bound()
        self.runtime.to(self.handle, "cpu", model=True, optimizer=False, grad=True)

    @torch.no_grad()
    def onload_model(self, models, onload_grad=True, tag="") -> None:
        del models, onload_grad, tag
        self._require_bound()
        self.runtime.to(self.handle, "cuda", model=True, optimizer=False, grad=True)

    @torch.no_grad()
    def offload_optimizer(self, optimizers) -> None:
        del optimizers
        self._require_bound()
        self.runtime.to(self.handle, "cpu", model=False, optimizer=True, grad=False)

    @torch.no_grad()
    def onload_optimizer(self, optimizers) -> None:
        del optimizers
        self._require_bound()
        self.runtime.to(self.handle, "cuda", model=False, optimizer=True, grad=False)
