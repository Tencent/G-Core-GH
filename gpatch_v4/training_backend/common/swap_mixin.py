from types import SimpleNamespace


class EngineSwapMixin:
    def get_swap_state(self) -> SimpleNamespace:
        if not hasattr(self, '_swap_state'):
            self._swap_state = SimpleNamespace(
                model=True,
                ref_model=True,
                optimizer=True,
            )
        return self._swap_state

    def offload_optimizer(self):
        assert self.swap_impl is not None
        if not self.get_swap_state().optimizer:
            return
        self.swap_impl.offload_optimizer(self.optimizer)
        self.get_swap_state().optimizer = False

    def offload_model(self):
        assert self.swap_impl is not None
        if not self.get_swap_state().model:
            return
        self.swap_impl.offload_model(self.model)
        if hasattr(self, 'ema_model') and self.ema_model is not None:
            self.swap_impl.offload_model(self.ema_model)
        self.get_swap_state().model = False

    def release_grad(self):
        assert self.swap_impl is not None
        self.swap_impl.release_grad(self.model)

    def onload_optimizer(self):
        assert self.swap_impl is not None
        if self.get_swap_state().optimizer:
            return
        self.swap_impl.onload_optimizer(self.optimizer)
        self.get_swap_state().optimizer = True

    def onload_model(self):
        assert self.swap_impl is not None
        if self.get_swap_state().model:
            return
        self.swap_impl.onload_model(self.model)
        if hasattr(self, 'ema_model') and self.ema_model is not None:
            self.swap_impl.onload_model(self.ema_model)
        self.get_swap_state().model = True

    def offload_ref_model(self):
        assert self.swap_impl is not None
        if self.ref_model is None:
            return
        if not self.get_swap_state().ref_model:
            return
        self.swap_impl.offload_model(self.ref_model)
        self.get_swap_state().ref_model = False

    def onload_ref_model(self):
        assert self.swap_impl is not None
        if self.ref_model is None:
            return
        if self.get_swap_state().ref_model:
            return
        self.swap_impl.onload_model(self.ref_model, onload_grad=False)
        self.get_swap_state().ref_model = True
