"""CLI entry point for training PPA on PARA GIAA.

Usage:
    PYTHONPATH=src uv run python scripts/train.py [--epochs N] [--batch-size N]
"""

import argparse
from pathlib import Path

from ppa.config import REPO_ROOT, Config, ScoreConfig, TrainConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    # Ablation: learn a temperature on the score sigmoid (PLCC-gap probe). Off by
    # default so the plain run stays a faithful reproduction.
    p.add_argument("--learnable-temp", action="store_true")
    p.add_argument("--ckpt-dir", type=str, default=None, help="overrides the default checkpoint dir")
    args = p.parse_args()

    base = TrainConfig()
    train_cfg = TrainConfig(
        lr=base.lr,
        betas=base.betas,
        batch_size=args.batch_size or base.batch_size,
        n_epochs=args.epochs or base.n_epochs,
        num_workers=base.num_workers,
        prompt_dropout=base.prompt_dropout,
        score_min=base.score_min,
        score_max=base.score_max,
    )
    cfg_kwargs = dict(train=train_cfg)
    if args.learnable_temp:
        cfg_kwargs["score"] = ScoreConfig(learnable_logit_scale=True)
        # Keep the faithful reproduction's checkpoints/ intact; ablation writes here.
        cfg_kwargs["ckpt_dir"] = REPO_ROOT / "checkpoints_learnable_temp"
    if args.ckpt_dir:
        cfg_kwargs["ckpt_dir"] = Path(args.ckpt_dir)
    if args.device:
        cfg_kwargs["device"] = args.device
    cfg = Config(**cfg_kwargs)

    # Imported here so --help stays fast (avoids importing torch/open_clip).
    from ppa.train import train_ppa

    train_ppa(cfg)


if __name__ == "__main__":
    main()
