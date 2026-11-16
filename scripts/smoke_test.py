"""Shape/smoke test for the PPA core (no data needed).

Builds the model, runs a forward pass on random images, and checks that the
prediction and NLL loss have the expected shapes and ranges.
"""

import torch

from ppa.config import Config
from ppa.model import PPA
from ppa.prompts import get_prompt_pairs


def main() -> None:
    cfg = Config(device="cpu")  # CPU is enough for a shape check
    pairs = get_prompt_pairs()
    print(f"prompt pairs: {len(pairs)}")

    model = PPA(cfg, pairs).to(cfg.device)
    model.train()

    b = 4
    images = torch.randn(b, 3, 224, 224, device=cfg.device)
    task_ids = torch.zeros(b, dtype=torch.long, device=cfg.device)
    target = torch.rand(b, device=cfg.device)  # in (0, 1)

    pred = model.predict(images, task_ids)
    loss = model.nll_loss(images, task_ids, target)

    print(f"text_diff: {tuple(model.text_diff.shape)}")
    print(f"pred: shape={tuple(pred.shape)} min={pred.min():.3f} max={pred.max():.3f}")
    print(f"nll loss: {loss.item():.4f}")

    loss.backward()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_with_grad = sum(
        p.numel() for p in model.parameters() if p.requires_grad and p.grad is not None
    )
    print(f"trainable params: {n_trainable:,}  (received grad: {n_with_grad:,})")

    assert pred.shape == (b,)
    assert (pred >= 0).all() and (pred <= 1).all()
    assert torch.isfinite(loss)
    assert n_with_grad > 0
    print("OK")


if __name__ == "__main__":
    main()
