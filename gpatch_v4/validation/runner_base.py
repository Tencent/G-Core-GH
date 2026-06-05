"""Shared base class for in-training validation runners."""

import gc
import json
import math
import os
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist

from .clip_metrics import CLIPMetrics
from .fsdp_utils import (
    register_fsdp_forward_methods,
    restore_model_for_training,
    snapshot_module_modes,
)


class BaseValidationRunner(ABC):
    """Shared orchestration for validation executed during training."""

    PHASE_ORDER = ("t2i", "edit", "vqa", "text")
    GENERATION_PHASES = ("t2i", "edit")

    def __init__(self, engine, training_args, logger):
        self.engine = engine
        self.training_args = training_args
        self.logger = logger
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.cuda.current_device()

        self.val_groups = self._load_val_data()
        self._clip_metrics = None

    @property
    def clip_metrics(self):
        if self._clip_metrics is None:
            self._clip_metrics = CLIPMetrics(device=self.device)
        return self._clip_metrics

    def _load_default_val_data(self) -> dict:
        val_path = self.training_args.validation_data_dir
        if not val_path:
            return {}

        if os.path.isfile(val_path):
            jsonl_path = val_path
        elif os.path.isdir(val_path):
            candidates = [name for name in os.listdir(val_path) if name.endswith(".jsonl")]
            if not candidates:
                return {}
            jsonl_path = os.path.join(val_path, candidates[0])
        else:
            self.logger.info(f"[Validation] path not found: {val_path}")
            return {}

        with open(jsonl_path, "r", encoding="utf-8") as f:
            raw_cases = [json.loads(line) for line in f if line.strip()]

        if self.rank == 0:
            self.logger.info(f"[Validation] Loaded {len(raw_cases)} cases from {jsonl_path}")

        groups = self._group_cases_by_phase(raw_cases)
        if self.rank == 0:
            for key, value in groups.items():
                if value:
                    self.logger.info(f"[Validation]   {key}: {len(value)} cases")

        return {key: value for key, value in groups.items() if value}

    @staticmethod
    def _group_cases_by_phase(raw_cases):
        groups = {"t2i": [], "edit": [], "vqa": [], "text": []}
        for case in raw_cases:
            scenario = case.get("scenario", "")
            task_type = case.get("task_type", "t2i")
            if "纯文本理解" in scenario:
                groups["text"].append(case)
            elif "理解" in scenario:
                groups["vqa"].append(case)
            elif task_type == "edit":
                groups["edit"].append(case)
            else:
                groups["t2i"].append(case)
        return groups

    def _build_save_root(self, curr_step: int) -> str:
        base_dir = getattr(self.training_args, "validation_save_dir", "")
        if not base_dir:
            base_dir = os.path.join(self.training_args.checkpoint_dir, "val_results")
        return os.path.join(base_dir, f"step_{curr_step:07d}")

    def _pad_and_shard(self, cases):
        padded_n = math.ceil(len(cases) / self.world_size) * self.world_size
        padded = cases + [None] * (padded_n - len(cases))
        per_rank = padded_n // self.world_size
        return padded, per_rank

    def _cleanup_after_run(self):
        if self._clip_metrics is not None:
            self._clip_metrics.cleanup()
            self._clip_metrics = None

        gc.collect()
        torch.cuda.empty_cache()

    def run(self, curr_step: int) -> tuple:
        if not self.val_groups:
            return {}, ""

        metrics = {}
        save_root = self._build_save_root(curr_step)
        os.makedirs(save_root, exist_ok=True)
        summary_path = os.path.join(save_root, "metrics.json")

        for model, tag in ((self.engine.fsdp_model, "train"), (self.engine.ema_model, "ema")):
            if model is None:
                continue

            register_fsdp_forward_methods(model)
            mode_snapshot = snapshot_module_modes(model)
            model.eval()
            try:
                model_dir = os.path.join(save_root, tag)
                os.makedirs(model_dir, exist_ok=True)
                dist.barrier()

                model_metrics = self._validate_model(model, tag, model_dir)
                metrics.update(model_metrics)

                if self.rank == 0:
                    with open(summary_path, "w", encoding="utf-8") as f:
                        json.dump(metrics, f, indent=2, ensure_ascii=False)
            finally:
                restore_model_for_training(model, mode_snapshot)

        self._cleanup_after_run()
        dist.barrier()

        if self.rank == 0:
            self.logger.info(f"[Validation] Metrics saved to {summary_path}")

        return metrics, save_root

    def _validate_model(self, model, model_tag: str, save_dir: str) -> dict:
        inferencer = self.build_inferencer(model)
        metrics = {}
        for phase_name in self.PHASE_ORDER:
            cases = self.val_groups.get(phase_name, [])
            if not cases:
                continue
            metrics.update(self._run_phase(inferencer, cases, phase_name, model_tag, save_dir))
        return metrics

    def _run_phase(self, inferencer, cases, phase_name: str, model_tag: str, save_dir: str) -> dict:
        is_gen = phase_name in self.GENERATION_PHASES
        padded_cases, per_rank = self._pad_and_shard(cases)
        start = self.rank * per_rank
        my_cases = padded_cases[start:start + per_rank]

        clip_scores = []
        ref_similarities = []
        und_results = []
        total_cases = len(cases)
        num_my_cases = len(my_cases)

        for case_idx, case in enumerate(my_cases):
            is_dummy = case is None
            if is_dummy:
                case = self.make_dummy_case(phase_name)

            images, ref_images = self.load_case_images(case)
            config = self.build_infer_config(case, phase_name)

            try:
                result = inferencer(
                    image=images if images else None,
                    text=case.get("text") or None,
                    **config,
                )
            except Exception as exc:
                if self.rank == 0:
                    name = case.get("name", "?")
                    self.logger.info(f"[Validation] {phase_name} error ({name}): {exc}")
                    import traceback
                    traceback.print_exc()
                result = {"image": None, "text": None}

            if self.rank == 0:
                self.logger.info(
                    f"[Validation] {model_tag}/{phase_name}: "
                    f"case {case_idx + 1}/{num_my_cases} done (total {total_cases} cases across {self.world_size} ranks)"
                )

            if is_dummy:
                continue

            name = case.get("name", f"{phase_name}_unknown")
            if is_gen and result.get("image") is not None:
                result["image"].save(os.path.join(save_dir, f"{name}.png"))

                try:
                    text = case.get("text", "")
                    if text:
                        clip_scores.append(
                            self.clip_metrics.text_image_score(text, result["image"])
                        )
                    if ref_images:
                        sims = [
                            self.clip_metrics.image_image_score(result["image"], ref)
                            for ref in ref_images
                        ]
                        ref_similarities.append(sum(sims) / len(sims))
                except Exception as exc:
                    if self.rank == 0:
                        self.logger.info(f"[Validation] CLIP score error ({name}): {exc}")
            elif not is_gen and result.get("text") is not None:
                und_results.append(
                    {
                        "name": name,
                        "prompt": case.get("text", ""),
                        "generated": result["text"],
                    }
                )

        if und_results:
            out_path = os.path.join(save_dir, f"{phase_name}_results_rank{self.rank}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(und_results, f, ensure_ascii=False, indent=2)

        return self._aggregate_phase_metrics(
            phase_name=phase_name,
            model_tag=model_tag,
            clip_scores=clip_scores,
            ref_similarities=ref_similarities,
            und_results=und_results,
        )

    def _aggregate_phase_metrics(
        self, phase_name, model_tag, clip_scores, ref_similarities, und_results
    ):
        metrics = {}
        if phase_name in self.GENERATION_PHASES:
            local_clip_sum = torch.tensor(
                sum(clip_scores) if clip_scores else 0.0, device=self.device
            )
            local_clip_cnt = torch.tensor(len(clip_scores), device=self.device)
            dist.all_reduce(local_clip_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_clip_cnt, op=dist.ReduceOp.SUM)
            avg_clip = (local_clip_sum /
                        local_clip_cnt).item() if local_clip_cnt.item() > 0 else 0.0
            metrics[f"val/{model_tag}/{phase_name}_clip_score"] = avg_clip

            if phase_name == "edit":
                local_ref_sum = torch.tensor(
                    sum(ref_similarities) if ref_similarities else 0.0,
                    device=self.device,
                )
                local_ref_cnt = torch.tensor(len(ref_similarities), device=self.device)
                dist.all_reduce(local_ref_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(local_ref_cnt, op=dist.ReduceOp.SUM)
                avg_ref = (local_ref_sum /
                           local_ref_cnt).item() if local_ref_cnt.item() > 0 else 0.0
                metrics[f"val/{model_tag}/{phase_name}_ref_similarity"] = avg_ref

            if self.rank == 0:
                joined = ", ".join(
                    f"{key.split('/')[-1]}={value:.4f}" for key, value in metrics.items()
                )
                self.logger.info(f"[Validation] {model_tag}/{phase_name}: {joined}")
        else:
            total_tensor = torch.tensor(len(und_results), device=self.device)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
            metrics[f"val/{model_tag}/{phase_name}_count"] = total_tensor.item()
            if self.rank == 0:
                self.logger.info(
                    f"[Validation] {model_tag}/{phase_name}: {total_tensor.item()} text results saved"
                )

        return metrics

    def _load_val_data(self) -> dict:
        """Load validation data and return grouped cases."""
        return self._load_default_val_data()

    @abstractmethod
    def build_inferencer(self, model):
        """Build the model-specific inferencer used during validation."""

    @abstractmethod
    def build_infer_config(self, case, phase_name: str) -> dict:
        """Build inferencer kwargs for a validation case."""

    @abstractmethod
    def load_case_images(self, case):
        """Return (input_images, reference_images) for a validation case."""

    @abstractmethod
    def make_dummy_case(self, phase_name: str) -> dict:
        """Return a phase-compatible dummy case for distributed padding."""
