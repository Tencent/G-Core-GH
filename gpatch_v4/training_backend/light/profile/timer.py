from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Optional

import torch


class DeviceTimer:
    """A lightweight event factory for accurate GPU timing.

    DeviceTimer only creates events and calls ``record()`` on them.
    It does **not** store event pairs, synchronize, or compute
    elapsed times — all of that is handled by :class:`TimerManager`.

    Example:
        >>> timer = DeviceTimer(device_module=torch.cuda)
        >>> start_ev = timer.start()
        >>> # ... GPU operations ...
        >>> end_ev = timer.end()
    """
    def __init__(self, enable: bool = True, device_module=None):
        self._enable = enable
        # Allow injecting a device module (e.g. torch.cuda, torch_npu, torch.mps).
        # Falls back to torch.cuda if not specified.
        self._device_module = device_module or torch.cuda

    @property
    def enabled(self) -> bool:
        return self._enable and self._device_module.is_available()

    def new_event(self):
        """Create a new device event with timing enabled."""
        return self._device_module.Event(enable_timing=True)

    def start(self):
        """Create and record a start event.

        Returns:
            The recorded start event, or None if disabled.
        """
        if not self.enabled:
            return None
        self.start_event = self.new_event()
        self.start_event.record()
        return self

    def end(self):
        """Create and record an end event.

        Returns:
            The recorded end event, or None if disabled.
        """
        if not self.enabled:
            return None
        self.end_event = self.new_event()
        self.end_event.record()
        return self

    def reset(self):
        self.start_event = None
        self.end_event = None

    def elapsed_time(self):
        if not self.enabled:
            return 0.0
        if self.start_event is None or self.end_event is None:
            return 0.0
        return self.start_event.elapsed_time(self.end_event)


class TimerManager:
    """Simple manager that holds named CudaTimers and provides unified reporting.

    Uses a :class:`DeviceTimer` internally as an event factory.
    All event-pair storage, device synchronization, and elapsed-time
    computation happen here — **never** inside ``DeviceTimer``.

    Example:
        >>> mgr = TimerManager.get_instance(device_module=torch.cuda)
        >>> with mgr.region("forward"):
        ...     out = model(x)
        >>> with mgr.region("backward"):
        ...     loss.backward()
        >>> mgr.summary()
    """

    _instance: Optional["TimerManager"] = None

    def __init__(
        self,
        enable: bool = True,
        device_module=None,
    ):
        self._enable = enable
        self._device_module = device_module
        self._timers = defaultdict(
            lambda: DeviceTimer(enable=self._enable, device_module=self._device_module)
        )
        self.records = defaultdict(list)

    # ------------------------------------------------------------------ #
    #  Global singleton
    # ------------------------------------------------------------------ #
    @classmethod
    def get_instance(
        cls,
        enable: bool = True,
        device_module=None,
    ) -> "TimerManager":
        """Return (and optionally create) the global TimerManager singleton."""
        if cls._instance is None:
            cls._instance = cls(
                enable=enable,
                device_module=device_module,
            )
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Destroy the global singleton (useful for testing)."""
        cls._instance = None

    def get_or_create_timer(self, name: str) -> DeviceTimer:
        timer = self._timers[name]
        # reset timer
        timer.reset()
        return timer

    # ------------------------------------------------------------------ #
    #  Recording helpers
    # ------------------------------------------------------------------ #
    @contextmanager
    def region(self, name: str):
        """Context manager that times a named region.

        Creates event pairs via the internal DeviceTimer and stores them
        in ``self._events``.

        Args:
            name: Region name for summary reporting.

        Example:
            >>> with mgr.region("attention"):
            ...     out = attn(q, k, v)
        """
        t = self.get_or_create_timer(name)
        if not t.enabled:
            yield
            return
        start_event = t.start()
        try:
            yield
        finally:
            t.end()

    # ------------------------------------------------------------------ #
    #  Synchronize & Reporting
    # ------------------------------------------------------------------ #
    def synchronize(self):
        """Synchronize the device exactly ONCE."""
        dm = self._device_module or torch.cuda
        if dm.is_available():
            dm.synchronize()

        for region_name, timer in self._timers.items():
            self.records[region_name].append(timer.elapsed_time())
            timer.reset()

    def get_value(self, name):
        # 区间可能在某些配置下从不触发（如纯图像训练没有 audio_encode、无 inpaint 没有 vae_mask）。
        # 此时返回空列表而非 assert 崩溃；调用方已用 `vals[-1] if vals else 0.0` 兜底。
        return self.records.get(name, [])

    def reset(self):
        """Clear all stored event pairs."""
        self.records.clear()

    def __repr__(self) -> str:
        return f"TimerManager(timers={list(self._timers.keys())})"
