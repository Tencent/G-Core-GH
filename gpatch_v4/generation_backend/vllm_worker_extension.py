# copyright (c) 2025 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com

import os
from typing import Any

import torch

from gpatch_v4.generation_backend.vllm_model_specific import (
    finalize_weights_after_reload,
    load_weights_for_update,
    prepare_weights_for_reload,
    restore_moe_after_wakeup,
    save_moe_for_sleep,
)
from gpatch_v4.utils import log


class GCoreVllmWorkerExtension:
    """Worker extension mixed into vLLM's Worker via ``worker_extension_cls``.

    Methods on this class are invoked through ``AsyncLLM.collective_rpc``
    (which dispatches to every TP worker). ``self`` at call time is the
    concrete vLLM Worker instance, so attributes such as
    ``self.weight_transfer_engine``, ``self.model_runner``,
    ``self.model_config``, and ``self.device`` are available.
    """
    def gcore_start_weights_update(self) -> None:
        """Prepare the model to receive checkpoint-format weights.

        Restores MoE params and attaches weight loaders (verl-style, no
        vLLM layerwise reload). Must be called once before the first bucket
        of a multi-bucket update; pair with
        :meth:`gcore_finalize_weights_update` after the last bucket.
        """
        if getattr(self, "_gcore_weight_update_active", False):
            raise RuntimeError(
                "gcore_start_weights_update called while a weight update "
                "is already active. Call gcore_finalize_weights_update first."
            )

        model = self.model_runner.model
        with torch.device(self.device):
            prepare_weights_for_reload(model, self.model_runner, self.device)
        self._gcore_weight_update_active = True

    def gcore_update_weights_ipc(self, update_info: dict[str, Any]) -> None:
        """Per-bucket weight receive via native IPCWeightTransferEngine.

        Parameters
        ----------
        update_info : dict
            Backend-specific update dict (same shape as what vLLM's built-in
            ``Worker.update_weights`` accepts). ``is_last_bucket`` is ignored;
            the trainer must call :meth:`gcore_finalize_weights_update` once
            after all buckets are loaded.
        """
        if self.weight_transfer_engine is None:
            raise RuntimeError(
                "Weight transfer not configured. "
                "Please set weight_transfer_config to enable weight transfer."
            )

        if not getattr(self, "_gcore_weight_update_active", False):
            raise RuntimeError(
                "gcore_start_weights_update must be called before "
                "gcore_update_weights_ipc."
            )

        # Strip legacy control flag before handing the dict to vLLM's parser.
        update_info = dict(update_info)
        update_info.pop("is_last_bucket", None)

        typed_update_info = self.weight_transfer_engine.parse_update_info(update_info)
        model = self.model_runner.model

        with torch.device(self.device):
            self.weight_transfer_engine.receive_weights(
                typed_update_info,
                load_weights=lambda weights: load_weights_for_update(
                    model,
                    self.model_runner,
                    weights,
                ),
            )

    def gcore_update_weights_bucketed(self, update_info: dict[str, Any]) -> None:
        """Receive one dtype-homogeneous flat-IPC bucket and load it.

        ``update_info`` is the payload produced by
        :class:`FlatIpcBucketBuilder` + the trainer all_gather in
        ``VllmUpdateWeightFactory._update_bucketed_ipc``; see
        :func:`open_flat_ipc_bucket` for the exact schema.

        No finalize is invoked here — the trainer calls
        :meth:`gcore_finalize_weights_update` exactly once after all buckets.
        """
        from gpatch_v4.generation_backend.bucketed_ipc_transfer import (
            open_flat_ipc_bucket,
        )

        if not getattr(self, "_gcore_weight_update_active", False):
            raise RuntimeError(
                "gcore_start_weights_update must be called before "
                "gcore_update_weights_bucketed."
            )

        model = self.model_runner.model

        # ``flat`` is kept alive for the duration of ``load_weights`` so
        # the narrow views remain valid; the trainer won't release the
        # CUDA IPC segment until our Ray reply is acked (ray.get).
        flat, model_weights = open_flat_ipc_bucket(update_info, self.device)

        with torch.device(self.device):
            load_weights_for_update(model, self.model_runner, model_weights)
        del model_weights, flat

    def gcore_update_weights_distributed(self, update_info: dict[str, Any]) -> None:
        """Receive one flat NCCL-broadcast bucket and load it into the model.

        ``process_weights_after_loading`` is **not** called here — the
        trainer issues a single :meth:`gcore_finalize_weights_update`
        RPC after all buckets are acked.
        """
        if self.weight_transfer_engine is None:
            raise RuntimeError(
                "Weight transfer not configured. "
                "Please set weight_transfer_config to enable weight transfer."
            )

        names = update_info["names"]
        dtype_names = update_info["dtype_names"]
        shapes = update_info["shapes"]
        total_bytes = int(update_info["total_bytes"])

        group = self.weight_transfer_engine.model_update_group
        if group is None:
            raise RuntimeError(
                "NCCL weight transfer group not initialized. "
                "Call init_weight_transfer_engine() first."
            )

        flat = torch.empty(total_bytes, dtype=torch.uint8, device=self.device)
        group.broadcast(flat, src=0, stream=torch.cuda.current_stream())
        torch.cuda.current_stream().synchronize()

        weights: list = []
        offset = 0
        for name, dn, shape in zip(names, dtype_names, shapes):
            dtype = getattr(torch, dn)
            numel = 1
            for s in shape:
                numel *= s
            nbytes = numel * dtype.itemsize
            view = flat[offset:offset + nbytes].view(dtype).view(shape)
            weights.append((name, view))
            offset += nbytes

        assert offset == total_bytes, (
            f"distributed bucket byte layout mismatch: offset={offset} total_bytes={total_bytes}"
        )

        if not getattr(self, "_gcore_weight_update_active", False):
            raise RuntimeError(
                "gcore_start_weights_update must be called before "
                "gcore_update_weights_distributed."
            )

        model = self.model_runner.model
        with torch.device(self.device):
            load_weights_for_update(model, self.model_runner, weights)
        del weights, flat

    def gcore_finalize_weights_update(self) -> None:
        """Finalize one full checkpoint-format weight update.

        Runs MegaMoE finalize and vLLM ``process_weights_after_loading``.
        Called by the trainer exactly once after all buckets are loaded.
        """
        if not getattr(self, "_gcore_weight_update_active", False):
            raise RuntimeError(
                "gcore_finalize_weights_update called without an active "
                "weight update. Call gcore_start_weights_update first."
            )

        model = self.model_runner.model
        model_config = self.model_runner.vllm_config.model_config

        with torch.device(self.device):
            finalize_weights_after_reload(model, model_config, self.model_runner, self.device)
        self._gcore_weight_update_active = False

    def gcore_save_moe_for_sleep(self) -> None:
        stash = save_moe_for_sleep(self.model_runner.model, self.model_runner, self.device)
        self._gcore_moe_sleep_stash = stash

    def gcore_restore_moe_after_wakeup(self) -> None:
        stash = getattr(self, "_gcore_moe_sleep_stash", None)
        if stash:
            restore_moe_after_wakeup(self.model_runner.model, self.model_runner, self.device, stash)
            self._gcore_moe_sleep_stash = {}
