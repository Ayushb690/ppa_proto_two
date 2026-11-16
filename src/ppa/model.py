"""PPA model: ties p_theta and p_phi together (paper Eq. 2, Eq. 5, Eq. 6).

Core mixture-over-prompts model (Eq. 2):
    p(s | x, c) = sum_t  p_theta(s | x, t) * p_phi(t | x, c)

Training minimizes the negative log-likelihood of a Gaussian mixture (Eq. 5):
    L = - sum_i log sum_t exp( -(s_i - s_bar(x_i, t))^2 / (2 sigma^2) ) * p_phi(t | x_i, c_i)
(the Gaussian normalizer is a constant in sigma and dropped).

Inference takes the expectation (Eq. 6):
    E[s | x, c] = sum_t s_bar(x, t) * p_phi(t | x, c)
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .clip_backbone import CLIPBackbone
from .config import Config
from .prompts import PromptPair
from .score_model import clipiqa_score
from .selection_model import PromptSelectionModel


class PPA(nn.Module):
    def __init__(self, cfg: Config, prompt_pairs: list[PromptPair]):
        super().__init__()
        self.cfg = cfg
        self.backbone = CLIPBackbone(cfg.clip)
        self.selection = PromptSelectionModel(
            embed_dim=cfg.clip.embed_dim,
            task_embed_dim=cfg.selection.task_embed_dim,
            hidden_dim=cfg.selection.hidden_dim,
            proj_dim=cfg.selection.proj_dim,
            n_tasks=cfg.selection.n_tasks,
        )

        # Precompute the frozen text embeddings for the prompt pool once.
        pos = self.backbone.encode_text([p.pos for p in prompt_pairs])  # (P, D)
        neg = self.backbone.encode_text([p.neg for p in prompt_pairs])  # (P, D)
        self.register_buffer("pos_emb", pos)
        self.register_buffer("neg_emb", neg)
        self.register_buffer("text_diff", pos - neg)  # e_text per prompt (Sec. 3.3)

        # Ablation knob (off by default): a learnable temperature on the score
        # logit. Stored in log-space so exp() keeps it positive during training.
        if cfg.score.learnable_logit_scale:
            self.log_logit_scale = nn.Parameter(
                torch.tensor(math.log(cfg.score.logit_scale_init))
            )

    @property
    def _logit_scale(self) -> Tensor | float:
        if self.cfg.score.learnable_logit_scale:
            return self.log_logit_scale.exp()
        return self.backbone.logit_scale if self.cfg.score.use_logit_scale else 1.0

    def _scores_and_weights(
        self, images: Tensor, task_ids: Tensor, prompt_idx: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        # prompt_idx restricts the prompt pool to a subset (used for prompt-dropout).
        pos, neg, text_diff = self.pos_emb, self.neg_emb, self.text_diff
        if prompt_idx is not None:
            pos, neg, text_diff = pos[prompt_idx], neg[prompt_idx], text_diff[prompt_idx]

        image_emb = self.backbone.encode_image(images)                  # (B, D)
        s_bar = clipiqa_score(image_emb, pos, neg, self._logit_scale)   # (B, P')
        # p_phi is a softmax over whatever prompt subset is passed, so it
        # renormalizes automatically over the surviving prompts.
        p_phi = self.selection(image_emb, text_diff, task_ids)          # (B, P')
        return s_bar, p_phi

    def _dropout_idx(self, device: torch.device) -> Tensor | None:
        """Sample the surviving prompt indices for prompt-dropout (training only)."""
        rate = self.cfg.train.prompt_dropout
        if not self.training or rate <= 0.0:
            return None
        p = self.pos_emb.shape[0]
        n_keep = max(1, round(p * (1.0 - rate)))
        return torch.randperm(p, device=device)[:n_keep]

    def predict(self, images: Tensor, task_ids: Tensor) -> Tensor:
        """Eq. 6: expected score E[s | x, c], shape (B,), in (0, 1). Uses all prompts."""
        s_bar, p_phi = self._scores_and_weights(images, task_ids)
        return (s_bar * p_phi).sum(dim=-1)

    def nll_loss(self, images: Tensor, task_ids: Tensor, target: Tensor) -> Tensor:
        """Eq. 5: mean Gaussian-mixture NLL. target: (B,) in (0, 1)."""
        prompt_idx = self._dropout_idx(images.device)
        s_bar, p_phi = self._scores_and_weights(images, task_ids, prompt_idx)  # (B, P')
        sigma = self.cfg.score.sigma
        log_gauss = -((target.unsqueeze(-1) - s_bar) ** 2) / (2 * sigma**2)  # (B, P')
        # logsumexp over prompts of log p_phi + log Gaussian, for numerical stability.
        log_mix = torch.logsumexp(torch.log(p_phi + 1e-12) + log_gauss, dim=-1)  # (B,)
        return -log_mix.mean()
