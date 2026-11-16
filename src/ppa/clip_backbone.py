"""Frozen-ish CLIP backbone (paper Sec. 3.1, 3.3, 4.2).

We reuse OpenCLIP rather than reimplementing CLIP (it is frozen except for the
last few image-encoder blocks, so there is nothing to gain by self-implementing).

Trainability, per the paper:
  - Text encoder: fully frozen.
  - Image encoder: only the last ``n_unfrozen_visual_blocks`` transformer blocks
    are fine-tuned. These weights are the parameters ``theta`` of the score model.
"""

from __future__ import annotations

import open_clip
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import CLIPConfig


class CLIPBackbone(nn.Module):
    def __init__(self, cfg: CLIPConfig):
        super().__init__()
        self.cfg = cfg
        model, _, preprocess_val = open_clip.create_model_and_transforms(
            cfg.model_name, pretrained=cfg.pretrained
        )
        self.model = model
        self.preprocess = preprocess_val
        self.tokenizer = open_clip.get_tokenizer(cfg.model_name)

        self._set_trainable()

    def _set_trainable(self) -> None:
        """Freeze everything, then unfreeze the last N image-encoder blocks (theta)."""
        for p in self.model.parameters():
            p.requires_grad_(False)

        blocks = self.model.visual.transformer.resblocks
        n = self.cfg.n_unfrozen_visual_blocks
        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad_(True)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    @property
    def logit_scale(self) -> Tensor:
        # CLIP's learned temperature; exp() gives the scaling applied to cosine sims.
        return self.model.logit_scale.exp()

    def encode_image(self, images: Tensor) -> Tensor:
        """images: (B, 3, H, W) -> L2-normalized embeddings (B, D). Grad flows to theta."""
        feats = self.model.encode_image(images)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> Tensor:
        """texts: list[str] -> L2-normalized embeddings (len, D). Frozen, no grad."""
        device = next(self.model.parameters()).device
        tokens = self.tokenizer(texts).to(device)
        feats = self.model.encode_text(tokens)
        return F.normalize(feats, dim=-1)
