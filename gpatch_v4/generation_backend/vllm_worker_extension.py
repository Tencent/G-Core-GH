# copyright (c) 2025 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com

from typing import Any

import torch


class GCoreVllmWorkerExtension:
    """Worker extension mixed into vLLM's Worker via ``worker_extension_cls``.

    Methods on this class are invoked through ``AsyncLLM.collective_rpc``
    (which dispatches to every TP worker). ``self`` at call time is the
    concrete vLLM Worker instance, so attributes such as
    ``self.weight_transfer_engine``, ``self.model_runner``,
    ``self.model_config``, and ``self.device`` are available.
    """
    def gcore_update_weights_ipc(self, update_info: dict[str, Any]) -> None:
        """Per-bucket weight receive, with final post-load processing.

        Parameters
        ----------
        update_info : dict
            Backend-specific update dict (same shape as what vLLM's built-in
            ``Worker.update_weights`` accepts) with one extra key:

            - ``is_last_bucket`` (bool, optional, default ``True``):
              When ``True``, :func:`process_weights_after_loading` is run
              after the bucket is loaded. Trainer should set ``False`` for
              every bucket except the final one in a multi-bucket update.
        """
        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        if self.weight_transfer_engine is None:
            raise RuntimeError(
                "Weight transfer not configured. "
                "Please set weight_transfer_config to enable weight transfer."
            )

        # Strip our control flag before handing the dict to vLLM's parser,
        # which rejects unknown keys via dataclass kwargs.
        update_info = dict(update_info)
        is_last_bucket = bool(update_info.pop("is_last_bucket", True))

        typed_update_info = self.weight_transfer_engine.parse_update_info(update_info)
        model = self.model_runner.model

        with torch.device(self.device):
            self.weight_transfer_engine.receive_weights(
                typed_update_info,
                load_weights=model.load_weights,
            )
            if is_last_bucket:
                model_config = self.model_runner.vllm_config.model_config
                process_weights_after_loading(model, model_config, self.device)

    def gcore_update_weights_bucketed(self, update_info: dict[str, Any]) -> None:
        """Receive one dtype-homogeneous flat-IPC bucket and load it.

        ``update_info`` is the payload produced by
        :class:`FlatIpcBucketBuilder` + the trainer all_gather in
        ``UpdateWeightIpcMixin._update_weights_by_bucketed_ipc_vllm``; see
        :func:`open_flat_ipc_bucket` for the exact schema.

        We open the flat-IPC bucket via
        :func:`open_flat_ipc_bucket` (which handles uuid lookup, tensor
        rebuild with device-id retargeting, shape/dtype checks, and the
        ``narrow + view`` slicing) and feed the resulting ``(name, view)``
        pairs to ``model.load_weights``.

        No ``process_weights_after_loading`` is invoked here -- the
        trainer is expected to call :meth:`gcore_finalize_weights_update`
        exactly once at the end of the whole update.
        """
        from gpatch_v4.generation_backend.bucketed_ipc_transfer import open_flat_ipc_bucket
        from gpatch_v4.generation_backend.vllm_moe_weight_loader_patch import (
            patch_vllm_moe_model_weight_loader,
        )

        model = self.model_runner.model

        # Lazily apply the MoE weight_loader patch at the start of each
        # update. We key off a one-shot flag reset by the finalize RPC so
        # the patch runs exactly once per update (idempotent, but avoids
        # per-bucket layer iteration on large MoE models).
        if not getattr(self, "_gcore_bucketed_prepared", False):
            patch_vllm_moe_model_weight_loader(model)
            self._gcore_bucketed_prepared = True

        # ``flat`` is kept alive for the duration of ``load_weights`` so
        # the narrow views remain valid; the trainer won't release the
        # CUDA IPC segment until our Ray reply is acked (ray.get).
        flat, model_weights = open_flat_ipc_bucket(update_info, self.device)
        with torch.device(self.device):
            model.load_weights(weights=model_weights)
        del model_weights, flat

    def gcore_update_weights_distributed(self, update_info: dict[str, Any]) -> None:
        """Receive one flat NCCL-broadcast bucket and load it into the model.

        Sender-side counterpart:
        :meth:`UpdateWeightDistributedMixin._broadcast_weight_bucket_vllm`.

        ``update_info`` schema (all buckets share the same format):

        - ``names`` (list[str])
        - ``dtype_names`` (list[str]) -- e.g. ``["bfloat16", "float32"]``
        - ``shapes`` (list[list[int]])
        - ``total_bytes`` (int) -- flat uint8 bucket length
        - ``group_name`` (str) -- trainer-side NCCL group label (info only)

        Each worker allocates a contiguous uint8 tensor of ``total_bytes``,
        joins the ``PyNcclCommunicator.broadcast`` initiated by trainer
        rank 0 on the group created by
        :meth:`~.init_weight_transfer_engine`, then slices the flat buffer
        into per-tensor byte views and re-interprets them as
        ``view(dtype).view(shape)`` before handing them to
        ``model.load_weights``.

        ``process_weights_after_loading`` is **not** called here -- the
        trainer issues a single :meth:`gcore_finalize_weights_update`
        RPC after all buckets are acked to finalize the whole update
        (inlined at the tail of
        :meth:`UpdateWeightDistributedMixin._update_weights_by_distributed_vllm`).
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

        model = self.model_runner.model
        with torch.device(self.device):
            model.load_weights(weights=weights)
        # flat 在 load_weights 之后才释放，保证所有 view 段期间都有效。
        del weights, flat

    def gcore_finalize_weights_update(self) -> None:
        """Run ``process_weights_after_loading`` once per full weight update.

        Transport-agnostic finalize -- called by the trainer exactly once
        at the end of any multi-bucket weight sync path (bucketed-IPC or
        NCCL-distributed). Also resets the one-shot
        ``_gcore_bucketed_prepared`` marker so the next bucketed-IPC
        update re-applies the MoE weight-loader patch; setting the
        attribute is a no-op for non-bucketed paths (idempotent).
        """
        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        model = self.model_runner.model
        model_config = self.model_runner.vllm_config.model_config

        with torch.device(self.device):
            process_weights_after_loading(model, model_config, self.device)

        self._gcore_bucketed_prepared = False
