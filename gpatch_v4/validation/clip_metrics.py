"""CLIP-based metrics shared by Bagel / WGOv3 validation."""

import torch
import torch.nn.functional as F
from PIL import Image


class CLIPMetrics:
    """Text-image and image-image CLIP similarity scorer."""

    CLIP_LOCAL_PATH = "/mnt/shenzhen2cephfs/mm-base-vision/huangzp/pretrain/openai/clip-vit-large-patch14"

    def __init__(self, device: torch.device | str = "cuda"):
        self._device = torch.device(device)
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        if self._model is not None:
            return

        import logging
        import warnings

        import transformers
        from transformers import CLIPModel, CLIPProcessor

        # Suppress all noisy loading output (progress bar, LOAD REPORT, sharding warnings, etc.)
        _loggers_to_silence = [
            "transformers",
            "transformers.modeling_utils",
            "transformers.configuration_utils",
            "transformers.tokenization_utils_base",
            "transformers.image_processing_utils",
            "transformers.image_processing_utils_fast",
        ]
        _prev_levels = {n: logging.getLogger(n).level for n in _loggers_to_silence}
        _prev_verbosity = transformers.utils.logging.get_verbosity()
        _prev_progress = transformers.utils.logging.is_progress_bar_enabled()

        for n in _loggers_to_silence:
            logging.getLogger(n).setLevel(logging.ERROR)
        transformers.utils.logging.set_verbosity_error()
        transformers.utils.logging.disable_progress_bar()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._processor = CLIPProcessor.from_pretrained(
                    self.CLIP_LOCAL_PATH,
                    use_fast=False,
                )
                self._model = CLIPModel.from_pretrained(self.CLIP_LOCAL_PATH, ).to(self._device
                                                                                  ).eval()
        finally:
            for n, lvl in _prev_levels.items():
                logging.getLogger(n).setLevel(lvl)
            transformers.utils.logging.set_verbosity(_prev_verbosity)
            if _prev_progress:
                transformers.utils.logging.enable_progress_bar()

    @torch.no_grad()
    def text_image_score(self, text: str, image: Image.Image) -> float:
        self._ensure_loaded()
        inputs = self._processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        text_emb = F.normalize(outputs.text_embeds, dim=-1)
        image_emb = F.normalize(outputs.image_embeds, dim=-1)
        return (text_emb @ image_emb.T).item()

    @torch.no_grad()
    def image_image_score(self, img1: Image.Image, img2: Image.Image) -> float:
        self._ensure_loaded()
        inputs = self._processor(images=[img1, img2], return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        result = self._model.get_image_features(**inputs)
        # transformers v5: get_image_features returns BaseModelOutputWithPooling, not a Tensor.
        if isinstance(result, torch.Tensor):
            image_embs = result
        else:
            image_embs = result.pooler_output
        image_embs = F.normalize(image_embs, dim=-1)
        return (image_embs[0] @ image_embs[1]).item()

    def cleanup(self):
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        torch.cuda.empty_cache()
