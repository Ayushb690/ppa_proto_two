"""CLI entry point for training PPA on PARA GIAA.

Usage:
    uv run python train.py [--epochs N] [--batch-size N]
"""

import sys
from pathlib import Path

# Ensure src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ppa.config import Config, ScoreConfig, TrainConfig
import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--learnable-temp", action="store_true")
    p.add_argument("--ckpt-dir", type=str, default=None, help="overrides default checkpoint dir")
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
        cfg_kwargs["ckpt_dir"] = REPO_ROOT / "checkpoints_learnable_temp"
    if args.ckpt_dir:
        cfg_kwargs["ckpt_dir"] = Path(args.ckpt_dir)
    if args.device:
        cfg_kwargs["device"] = args.device
    cfg = Config(**cfg_kwargs)

    from ppa.train import train_ppa

    train_ppa(cfg)


if __name__ == "__main__":
    main()
