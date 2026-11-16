"""Central configuration for the PPA reproduction.

All hyperparameters are taken from the original paper (Hara & Otsuka, CVPR 2026,
"Probabilistic Prompt Adaptation for Unified Image Aesthetics and Quality Assessment"),
Sec. 3 and Sec. 4.2 (Implementation Details). Values are cited inline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

# Repository root: <repo>/src/ppa/config.py -> parents[2] == <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "PARA"


def resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass(frozen=True)
class CLIPConfig:
    # Paper: "We adopt CLIP-B/16 as the backbone network." (Sec. 4.2)
    model_name: str = "ViT-B-16"
    pretrained: str = "openai"
    # Paper: text encoder is frozen; only the last four blocks of the image
    # encoder are fine-tuned (these weights are the parameters theta).
    n_unfrozen_visual_blocks: int = 4
    embed_dim: int = 512  # ViT-B/16 CLIP joint embedding dimension D


@dataclass(frozen=True)
class SelectionConfig:
    """p_phi: prompt selection model (paper Eq. 3)."""

    # Paper: "the task embedding dimension is set to C = 4" (Sec. 4.2)
    task_embed_dim: int = 4
    # Paper: "passed through a 256-dimensional intermediate layer to produce a
    # final 128-dimensional embedding." (Sec. 4.2)
    hidden_dim: int = 256
    proj_dim: int = 128
    # Number of distinct evaluation tasks (datasets). For single-dataset PARA
    # GIAA reproduction this is 1; kept configurable for the multi-task setting.
    n_tasks: int = 1


@dataclass(frozen=True)
class ScoreConfig:
    """p_theta: Gaussian score model around the CLIP-IQA score (Eq. 1, Eq. 4)."""

    # Paper: "we set sigma = 0.1." (Sec. 4.2)
    sigma: float = 0.1
    # Eq. 1 uses the raw inner product of L2-normalized embeddings (i.e. cosine),
    # with NO temperature/logit scale -- confirmed absent from both the main paper
    # and the supplementary. We follow the paper literally (default off). The flag
    # is kept because CLIP-IQA conventionally applies the logit scale, which widens
    # the score range; useful as an ablation knob.
    use_logit_scale: bool = False
    # Ablation (default off, to stay faithful): a single learnable temperature on
    # the score logit, score = tau * (sim_pos - sim_neg). With tau = 1 the sigmoid
    # squashes predictions into a narrow band (~[0.35, 0.65]), which caps PLCC; a
    # larger tau widens them. Made learnable so the model finds the sharpness itself
    # instead of sweeping fixed values. Initialized below.
    learnable_logit_scale: bool = False
    logit_scale_init: float = 10.0


@dataclass(frozen=True)
class TrainConfig:
    # Paper: "Adam optimizer ... learning rate of 1e-5 and beta values of
    # 0.9 and 0.999." (Sec. 4.2)
    lr: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.999)
    batch_size: int = 32
    n_epochs: int = 20
    num_workers: int = 8
    # Paper supplementary (Sec. B): "prompt-dropout regularization during training
    # by randomly disabling 50% of prompts per iteration to reduce reliance on any
    # fixed prompt set." Applied to the marginalization / selection side only.
    prompt_dropout: float = 0.5
    # PARA aesthetic scores live in [1, 5]; the CLIP-IQA score s_bar lives in
    # (0, 1). We map targets to [0, 1] via (s - 1) / 4 so the NLL's squared term
    # is on a comparable scale. TODO: confirm normalization against supplementary.
    score_min: float = 1.0
    score_max: float = 5.0


@dataclass(frozen=True)
class Config:
    data_root: Path = DATA_ROOT
    clip: CLIPConfig = field(default_factory=CLIPConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = field(default_factory=resolve_device)
    seed: int = 42
    # Checkpoints (git-ignored). best.pt tracks the highest validation SRCC;
    # last.pt is the most recent epoch, for resuming a long run that died.
    ckpt_dir: Path = REPO_ROOT / "checkpoints"
