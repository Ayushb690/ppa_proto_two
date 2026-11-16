"""p_theta: the CLIP-IQA score s_bar(x, t) (paper Eq. 1, Eq. 4).

p_theta(s | x, t) is a Gaussian whose mean is the CLIP-IQA score and whose
variance sigma^2 is a fixed hyperparameter (Eq. 4). The mean has NO dedicated
parameters of its own: theta is exactly the trainable last blocks of the image
encoder. So this module is parameter-free; it just evaluates Eq. 1.

Eq. 1 (per antonym pair t = (t_pos, t_neg)):

    s_bar(x, t) = softmax over {pos, neg} of cosine sims
                = sigmoid( w * ( <I(x), T(t_pos)> - <I(x), T(t_neg)> ) )

where w is the CLIP logit scale (temperature). Embeddings are L2-normalized, so
the inner product is the cosine similarity.
"""

from __future__ import annotations

import torch
from torch import Tensor


def clipiqa_score(
    image_emb: Tensor,   # (B, D), L2-normalized
    pos_emb: Tensor,     # (P, D), L2-normalized (frozen)
    neg_emb: Tensor,     # (P, D), L2-normalized (frozen)
    logit_scale: Tensor | float = 1.0,
) -> Tensor:
    """Return s_bar in (0, 1) of shape (B, P): the score per image per prompt pair."""
    sim_pos = image_emb @ pos_emb.t()   # (B, P)
    sim_neg = image_emb @ neg_emb.t()   # (B, P)
    return torch.sigmoid(logit_scale * (sim_pos - sim_neg))
