# copyright (c) 2024 tencent inc. all rights reserved.
# guanyouhe@tencent.com

"""WebDataset sample decoding utilities for Megatron Energon."""

from __future__ import annotations

import io
import json
from math import gcd
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from megatron.energon import CrudeSample, SkipSample

_IMAGE_EXTS = ("jpg", "jpeg", "png", "webp", "bmp")
_AUDIO_EXTS = ("wav", "flac", "mp3", "ogg", "opus")
_VIDEO_EXTS = ("mp4", "webm", "avi", "mkv")


def decode_json_by_wds(sample: CrudeSample) -> Dict[str, Any]:
    """Decode the ``json`` member of an Energon ``CrudeWebDataset`` sample."""
    if "json" not in sample or sample["json"] is None:
        raise SkipSample()
    meta = sample["json"]
    if isinstance(meta, (bytes, bytearray)):
        meta = json.loads(meta.decode("utf-8"))
    elif isinstance(meta, str):
        meta = json.loads(meta)
    elif not isinstance(meta, dict):
        meta = dict(meta)
    return meta


def _decode_image(value: Any) -> Optional[Any]:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    # Energon SampleDecoder may already yield CHW float/uint8 tensors.
    if isinstance(value, torch.Tensor):
        t = value.detach().cpu()
        if t.ndim == 3 and t.shape[0] in (1, 3, 4):
            t = t.permute(1, 2, 0)
        if t.ndim != 3:
            return None
        if t.dtype.is_floating_point:
            arr = (t.clamp(0, 1) * 255.0).to(torch.uint8).numpy()
        else:
            arr = t.to(torch.uint8).numpy()
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] == 4:
            arr = arr[..., :3]
        return Image.fromarray(arr, mode="RGB")
    if isinstance(value, np.ndarray):
        arr = value
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim != 3:
            return None
        if np.issubdtype(arr.dtype, np.floating):
            arr = (np.clip(arr, 0, 1) * 255.0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] == 4:
            arr = arr[..., :3]
        return Image.fromarray(arr, mode="RGB")
    blob = None
    if isinstance(value, (bytes, bytearray, memoryview)):
        blob = bytes(value)
    elif hasattr(value, "tobytes") and not isinstance(value, (str, dict)):
        try:
            blob = value.tobytes()
        except Exception:  # noqa: BLE001
            return None
    if blob is None:
        return None
    try:
        return Image.open(io.BytesIO(blob)).convert("RGB")
    except Exception:  # noqa: BLE001
        return None


def _decode_audio_bytes(blob: bytes, sr_target: Optional[int]) -> Optional[np.ndarray]:
    try:
        import soundfile as sf
    except ImportError as exc:  # noqa: BLE001
        raise ImportError(
            "decoding embedded audio requires soundfile; pip install soundfile"
        ) from exc
    try:
        audio_data, sr = sf.read(io.BytesIO(blob), dtype="float32")
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to decode embedded audio: {exc}")
        return None
    if sr_target is not None and sr != sr_target:
        try:
            from scipy.signal import resample_poly
        except ImportError as exc:  # noqa: BLE001
            raise ImportError(
                "audio resampling requires scipy; pip install scipy"
            ) from exc
        g = gcd(sr, sr_target)
        audio_data = resample_poly(
            audio_data, sr_target // g, sr // g, axis=0
        ).astype(np.float32)
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=-1)
    return audio_data


def _avdata_to_waveform(value: Any, sr_target: Optional[int]) -> Optional[np.ndarray]:
    if isinstance(value, np.ndarray):
        wav = value.astype(np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        return wav
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _decode_audio_bytes(bytes(value), sr_target)
    if hasattr(value, "tobytes") and not isinstance(value, (str, dict)):
        try:
            return _decode_audio_bytes(value.tobytes(), sr_target)
        except Exception:  # noqa: BLE001
            return None
    return None


def _collect_by_prefix(
    sample: CrudeSample,
    prefix: str,
    exts: tuple,
    decode_fn: Callable[[Any], Optional[Any]],
    skip_on_decode_fail: bool = True,
) -> List[Any]:
    indexed = []
    failed_keys = []
    for key, value in sample.items():
        if key.startswith("__") or value is None:
            continue
        if key.startswith(prefix) or key in exts:
            if isinstance(value, (dict, list)):
                continue
            decoded = decode_fn(value)
            if decoded is None:
                failed_keys.append(key)
            else:
                indexed.append((key, decoded))
    if failed_keys and skip_on_decode_fail:
        raise SkipSample()
    indexed.sort(key=lambda kv: kv[0])
    return [v for _, v in indexed]


def collect_media_by_wds(
    sample: CrudeSample,
    sr_target: Optional[int] = None,
) -> Dict[str, List[Any]]:
    """Collect media members from an Energon ``CrudeWebDataset`` sample."""
    media: Dict[str, List[Any]] = {}
    images = _collect_by_prefix(sample, "image", _IMAGE_EXTS, _decode_image)
    if images:
        media["__images_feat__"] = images
    audios = _collect_by_prefix(
        sample, "audio", _AUDIO_EXTS, lambda v: _avdata_to_waveform(v, sr_target)
    )
    if audios:
        media["__audios_feat__"] = audios

    def _decode_video(value: Any) -> Optional[Any]:
        if isinstance(value, torch.Tensor) and value.ndim == 4:
            return value
        return None

    videos = _collect_by_prefix(
        sample, "video", _VIDEO_EXTS, _decode_video, skip_on_decode_fail=False
    )
    if videos:
        media["__videos_feat__"] = videos
    return media
