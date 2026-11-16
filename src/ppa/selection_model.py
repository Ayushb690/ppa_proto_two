"""p_phi: the prompt selection model (paper Eq. 3, Sec. 3.3).

Given an image x and task c, it computes a distribution over the prompt pool
T_samp:

    p_phi(t | x, c) = softmax_t( f_phi1(x, c)^T  g_phi2(t, c) )

with embeddings (Sec. 3.3, 4.2):
    e_image = I(x)                 in R^D     (from the image encoder)
    e_text  = T(t_pos) - T(t_neg)  in R^D     (frozen text encoder, per prompt)
    e_task  = e_c                  in R^C     (learnable, C = 4)

Two two-layer MLPs map the concatenations into a shared R^{proj_dim} space:
    f_phi1: [e_image ; e_task] -> e_image_task   (B, proj_dim)
    g_phi2: [e_text  ; e_task] -> e_text_task    (B, P, proj_dim)

The prompt logits are the inner products of these projections.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _two_layer_mlp(in_dim: int, hidden: int, out: int) -> nn.Sequential:
    # Paper specifies a 256-d intermediate and a 128-d output; the nonlinearity
    # is unspecified, so we use ReLU. TODO: confirm against supplementary.
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, out),
    )


class PromptSelectionModel(nn.Module):
    def __init__(self, embed_dim: int, task_embed_dim: int, hidden_dim: int,
                 proj_dim: int, n_tasks: int):
        super().__init__()
        # Learnable task embedding e_c (paper Sec. 3.3).
        self.task_embed = nn.Embedding(n_tasks, task_embed_dim)
        self.f_phi1 = _two_layer_mlp(embed_dim + task_embed_dim, hidden_dim, proj_dim)
        self.g_phi2 = _two_layer_mlp(embed_dim + task_embed_dim, hidden_dim, proj_dim)

    def forward(
        self,
        image_emb: Tensor,   # (B, D), L2-normalized (grad flows to theta)
        text_diff: Tensor,   # (P, D) = T(t_pos) - T(t_neg), frozen
        task_ids: Tensor,    # (B,) long
    ) -> Tensor:
        """Return p_phi of shape (B, P): prompt selection probabilities."""
        b = image_emb.shape[0]
        p = text_diff.shape[0]

        e_task = self.task_embed(task_ids)                      # (B, C)

        # f branch: per image.
        f_in = torch.cat([image_emb, e_task], dim=-1)           # (B, D + C)
        f = self.f_phi1(f_in)                                   # (B, proj_dim)

        # g branch: per (image, prompt) because it is conditioned on the task too.
        text_diff_b = text_diff.unsqueeze(0).expand(b, p, -1)   # (B, P, D)
        e_task_b = e_task.unsqueeze(1).expand(b, p, -1)         # (B, P, C)
        g_in = torch.cat([text_diff_b, e_task_b], dim=-1)       # (B, P, D + C)
        g = self.g_phi2(g_in)                                   # (B, P, proj_dim)

        logits = torch.einsum("bd,bpd->bp", f, g)               # (B, P)
        return torch.softmax(logits, dim=-1)
