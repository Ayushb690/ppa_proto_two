"""Training loop for the PPA reproduction on PARA GIAA.

Optimizes the trainable parameters (last 4 image-encoder blocks = theta, and the
prompt selection model = phi) with Adam (lr 1e-5), minimizing the Gaussian-
mixture NLL (Eq. 5). Evaluation reports SRCC and PLCC against the ground-truth
mean aesthetic scores, the standard IAA metrics.

Checkpoint selection uses the held-out validation split (see ``data.py``); the
test split is evaluated exactly once, on the selected checkpoint, after training
ends. Selecting on test would report an optimistically biased number.
"""

from __future__ import annotations

import json

import torch
from torch import Tensor
from tqdm import tqdm

from .config import Config
from .data import build_dataloaders
from .model import PPA
from .prompts import get_prompt_pairs


def _pearson(x: Tensor, y: Tensor) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    return float((x @ y) / (denom + 1e-12))


def _spearman(x: Tensor, y: Tensor) -> float:
    # Spearman = Pearson on ranks. argsort().argsort() gives 0-based ranks.
    xr = x.argsort().argsort().float()
    yr = y.argsort().argsort().float()
    return _pearson(xr, yr)


@torch.no_grad()
def evaluate(model: PPA, loader, device: str) -> dict[str, float]:
    model.eval()
    preds, targets = [], []
    for images, task_ids, target in tqdm(loader, desc="eval", leave=False):
        images, task_ids = images.to(device), task_ids.to(device)
        preds.append(model.predict(images, task_ids).cpu())
        targets.append(target)
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    return {"srcc": _spearman(preds, targets), "plcc": _pearson(preds, targets)}


def train_ppa(cfg: Config) -> PPA:
    torch.manual_seed(cfg.seed)
    device = cfg.device

    model = PPA(cfg, get_prompt_pairs()).to(device)
    train_loader, val_loader, test_loader = build_dataloaders(cfg, model.backbone.preprocess)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=cfg.train.lr, betas=cfg.train.betas)
    print(f"device={device}  trainable params={sum(p.numel() for p in params):,}")

    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_srcc = -1.0
    best_epoch = 0

    for epoch in range(cfg.train.n_epochs):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.train.n_epochs}")
        for images, task_ids, target in pbar:
            images = images.to(device)
            task_ids = task_ids.to(device)
            target = target.to(device)

            loss = model.nll_loss(images, task_ids, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            pbar.set_postfix(nll=f"{loss.item():.4f}")

        metrics = evaluate(model, val_loader, device)
        avg = running / len(train_loader)
        print(
            f"[epoch {epoch + 1}] train_nll={avg:.4f}  "
            f"val_srcc={metrics['srcc']:.4f}  val_plcc={metrics['plcc']:.4f}"
        )

        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_metrics": metrics,
        }
        torch.save(ckpt, cfg.ckpt_dir / "last.pt")
        if metrics["srcc"] > best_srcc:
            best_srcc, best_epoch = metrics["srcc"], epoch + 1
            torch.save(ckpt, cfg.ckpt_dir / "best.pt")
            print(f"  -> new best (val_srcc={best_srcc:.4f}), saved best.pt")

    _report_test(model, cfg, test_loader, best_epoch)
    return model


def _report_test(model: PPA, cfg: Config, test_loader, best_epoch: int) -> None:
    """Evaluate the val-selected checkpoint on test, once, after training."""
    best = torch.load(cfg.ckpt_dir / "best.pt", map_location=cfg.device)
    model.load_state_dict(best["model"])
    test_metrics = evaluate(model, test_loader, cfg.device)
    print(
        f"[test @ epoch {best_epoch}] "
        f"test_srcc={test_metrics['srcc']:.4f}  test_plcc={test_metrics['plcc']:.4f}"
    )
    summary = {
        "selected_epoch": best_epoch,
        "val": best["val_metrics"],
        "test": test_metrics,
    }
    (cfg.ckpt_dir / "final_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
