"""Inference engine for the trained PPA (Probabilistic Prompt Adaptation) model.

Loads the PARA-trained checkpoint (default: Data Files/best.pt) and performs
deterministic, zero-overhead aesthetic assessment and prompt explainability
on input images.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from .config import Config, CLIPConfig, SelectionConfig, ScoreConfig, TrainConfig, REPO_ROOT
from .model import PPA
from .prompts import PromptPair, get_prompt_pairs, AESTHETIC_ATTRIBUTES, QUALITY_ATTRIBUTES
from .score_model import clipiqa_score

def _resolve_checkpoint_path(path: Path | str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    # Try alternate directory spellings (e.g. Data Files vs Data_files)
    candidates = [
        REPO_ROOT / "Data Files" / p.name,
        REPO_ROOT / "Data_files" / p.name,
        REPO_ROOT / "checkpoints" / p.name,
        Path.cwd() / "Data Files" / p.name,
        Path.cwd() / "Data_files" / p.name,
        REPO_ROOT / str(path).replace("Data_files", "Data Files"),
        REPO_ROOT / str(path).replace("Data Files", "Data_files"),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return p


DEFAULT_CKPT_PATH = _resolve_checkpoint_path(REPO_ROOT / "Data Files" / "best.pt")


class PPAInferenceEngine:
    """Production inference pipeline wrapping the trained PPA model."""

    def __init__(
        self,
        checkpoint_path: Path | str = DEFAULT_CKPT_PATH,
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"PPA checkpoint not found at: {checkpoint_path}")

        # Device determination: CUDA > CPU
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.device_name = self._get_device_display_name()

        # Load checkpoint dictionary
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.epoch = ckpt.get("epoch", 3)
        self.val_metrics = ckpt.get("val_metrics", {})

        # Reconstruct configuration
        cfg_dict = ckpt.get("config", {})
        self.cfg = Config(
            clip=CLIPConfig(**cfg_dict.get("clip", {})),
            selection=SelectionConfig(**cfg_dict.get("selection", {})),
            score=ScoreConfig(**cfg_dict.get("score", {})),
            train=TrainConfig(**cfg_dict.get("train", {})),
            device=self.device,
            seed=cfg_dict.get("seed", 42),
        )

        # Build prompt pool & model
        self.prompt_pairs: list[PromptPair] = get_prompt_pairs()
        self.model = PPA(self.cfg, self.prompt_pairs)

        # Load weights strictly
        load_result = self.model.load_state_dict(ckpt["model"], strict=True)
        self.missing_keys = load_result.missing_keys
        self.unexpected_keys = load_result.unexpected_keys

        # Move to target device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()

        self.preprocess = self.model.backbone.preprocess

    def _get_device_display_name(self) -> str:
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return f"CUDA:0 ({gpu_name})"
        return "CPU"

    def predict_image(
        self,
        image_input: Union[Image.Image, Path, str, bytes],
        task_id: int = 0,
    ) -> dict[str, Any]:
        """Perform end-to-end PPA prediction and concept explainability.

        Returns a dictionary conforming to the frontend's AnalysisResult interface.
        """
        start_time = time.perf_counter()

        # Load and convert image
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # Preprocess for CLIP ViT-B/16
        image_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        task_tensor = torch.tensor([task_id], dtype=torch.long, device=self.device)

        with torch.inference_mode():
            # Extract normalized image features
            image_emb = self.model.backbone.encode_image(image_tensor)
            
            # 1. CLIP-IQA scores s_bar across all 260 prompt pairs
            s_bar_tensor = clipiqa_score(
                image_emb, self.model.pos_emb, self.model.neg_emb, self.model._logit_scale
            )
            
            # 2. Strict model expected score (paper Eq. 6) using exact model selection
            p_phi_true = self.model.selection(image_emb, self.model.text_diff, task_tensor)
            norm_score = float((s_bar_tensor * p_phi_true).sum(dim=-1).item())
            norm_score = max(0.0, min(1.0, norm_score))
            raw_score = 1.0 + norm_score * 4.0  # Scale [0, 1] -> [1.0, 5.0]

            s_bar = s_bar_tensor[0].cpu().numpy()

            # 3. Compute selection logits z = f^T g for soft concept explainability
            e_task = self.model.selection.task_embed(task_tensor)
            f_in = torch.cat([image_emb, e_task], dim=-1)
            f = self.model.selection.f_phi1(f_in)
            b_size, p_size = 1, len(self.prompt_pairs)
            text_diff_b = self.model.text_diff.unsqueeze(0)
            e_task_b = e_task.unsqueeze(1).expand(b_size, p_size, -1)
            g_in = torch.cat([text_diff_b, e_task_b], dim=-1)
            g = self.model.selection.g_phi2(g_in)
            logits = torch.einsum("bd,bpd->bp", f, g)[0]

            # Temperature-scaled soft weights (tau = 15.0) for well-distributed explainability
            p_phi_soft = torch.softmax(logits / 15.0, dim=-1).cpu().numpy()

        inference_time_ms = round((time.perf_counter() - start_time) * 1000)

        # Tier classification
        tier, tier_color = self._classify_tier(raw_score)

        # Construct full prompt list with metrics
        all_prompts: list[dict[str, Any]] = []
        for idx, pair in enumerate(self.prompt_pairs):
            score_val = float(s_bar[idx])
            weight_val = float(p_phi_soft[idx])
            impact_val = score_val * weight_val
            all_prompts.append({
                "id": f"prompt-{idx}",
                "posPrompt": pair.pos,
                "negPrompt": pair.neg,
                "category": pair.category,
                "attribute": pair.attribute,
                "score": round(score_val, 4),
                "selectionWeight": round(weight_val, 6),
                "impact": round(impact_val, 6),
            })

        # Rank prompts:
        # Top positive drivers: highest positive impact
        top_positive = sorted(all_prompts, key=lambda p: p["impact"], reverse=True)[:6]
        # Top negative drivers: lowest score prompts with highest penalty
        low_score_prompts = [p for p in all_prompts if p["score"] < 0.6]
        if not low_score_prompts:
            low_score_prompts = sorted(all_prompts, key=lambda p: p["score"])[:10]
        # Sort negative prompts by penalty (low score and non-trivial weight)
        top_negative = sorted(low_score_prompts, key=lambda p: (1.0 - p["score"]) * p["selectionWeight"], reverse=True)[:5]

        # Attribute breakdowns
        attribute_breakdown = self._build_attribute_breakdown(
            all_prompts, AESTHETIC_ATTRIBUTES, norm_score
        )
        quality_breakdown = self._build_attribute_breakdown(
            all_prompts, QUALITY_ATTRIBUTES, norm_score
        )

        # Category weight distribution
        category_weights = self._compute_category_weights(all_prompts)

        return {
            "rawScore": round(raw_score, 2),
            "normalizedScore": round(norm_score, 4),
            "percentile": round(norm_score * 100),
            "tier": tier,
            "tierColor": tier_color,
            "inferenceTimeMs": max(1, inference_time_ms),
            "device": self.device_name,
            "topPositivePrompts": top_positive,
            "topNegativePrompts": top_negative,
            "allPrompts": all_prompts,
            "attributeBreakdown": attribute_breakdown,
            "qualityBreakdown": quality_breakdown,
            "categoryWeights": category_weights,
        }

    @staticmethod
    def _classify_tier(score: float) -> tuple[str, str]:
        if score >= 4.3:
            return "Masterpiece", "text-cyan-400 bg-cyan-400/10 border-cyan-400/30"
        if score >= 3.8:
            return "High Aesthetic", "text-emerald-400 bg-emerald-400/10 border-emerald-400/30"
        if score >= 3.2:
            return "Good Everyday", "text-amber-400 bg-amber-400/10 border-amber-400/30"
        if score >= 2.5:
            return "Sub-optimal", "text-orange-400 bg-orange-400/10 border-orange-400/30"
        return "Low / Flawed", "text-rose-400 bg-rose-400/10 border-rose-400/30"

    @staticmethod
    def _build_attribute_breakdown(
        all_prompts: list[dict[str, Any]],
        attributes: tuple[str, ...],
        fallback_norm: float,
    ) -> list[dict[str, Any]]:
        result = []
        for name in attributes:
            matching = [p for p in all_prompts if p["attribute"] == name]
            if matching:
                avg_score = sum(p["score"] for p in matching) / len(matching)
                sum_weight = sum(p["selectionWeight"] for p in matching)
            else:
                avg_score = fallback_norm
                sum_weight = 0.0

            result.append({
                "name": name,
                "score": round(avg_score * 100),
                "weight": round(sum_weight * 100, 1),
                "sampleCount": len(matching),
            })
        return result

    @staticmethod
    def _compute_category_weights(all_prompts: list[dict[str, Any]]) -> dict[str, int]:
        cat_sums = {
            "aesthetic_attribute": 0.0,
            "quality_attribute": 0.0,
            "aesthetic_general": 0.0,
            "quality_general": 0.0,
        }
        for p in all_prompts:
            cat = p.get("category", "")
            if cat in cat_sums:
                cat_sums[cat] += p["selectionWeight"]

        total = sum(cat_sums.values())
        if total <= 0.0:
            total = 1.0

        return {
            "aestheticAttribute": round((cat_sums["aesthetic_attribute"] / total) * 100),
            "qualityAttribute": round((cat_sums["quality_attribute"] / total) * 100),
            "aestheticGeneral": round((cat_sums["aesthetic_general"] / total) * 100),
            "qualityGeneral": round((cat_sums["quality_general"] / total) * 100),
        }
