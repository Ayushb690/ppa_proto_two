# `ppa` — PPA reproduction

Unofficial, from-scratch reimplementation of **Probabilistic Prompt Adaptation**
(Hara & Otsuka, *Probabilistic Prompt Adaptation for Unified Image Aesthetics and Quality
Assessment*, CVPR 2026), method only. This package reproduces the PPA scoring path and
training objective faithfully from the paper and its supplementary material.

> Not the authors' official code; not affiliated with or endorsed by them. Deviations are ours.

---

## Overview

PPA predicts an image aesthetic/quality score by **marginalizing over interpretable antonym
prompt pairs**. Each pair (e.g. *"beautifully composed photo"* ⇄ *"poorly composed photo"*)
acts as a fixed, human-readable concept. For an image, the model (a) scores it under every
prompt and (b) predicts how much each prompt should be weighted, then aggregates. Read as a
soft Concept Bottleneck Model: prompts are concepts, per-prompt scores are activations, and
the prompt weights are the concept weights.

---

## Architecture (paper Fig. 2)

```
                       ┌──────────── CLIP ────────────┐
  Image x ────────────►│ Image encoder  (last 4 blocks│──┬───────────────┐
                       │   trainable = theta)         │  │ I(x)          │ I(x)
   LLM (offline)       │                              │  ▼               ▼
    · prompt pool      │                              │ ┌────────────┐ ┌──────────────────────┐
  Text prompt t ──────►│ Text encoder   (frozen)      │►│ p_theta    │ │ p_phi                │◄─ task c
   (t_pos, t_neg)      └──────────────────────────────┘ │ score s̄    │ │ prompt selection     │
                                                  T(t)─►│ (Eq. 1)    │ │ (Eq. 3, trainable)   │
                                                        └─────┬──────┘ └──────────┬───────────┘
                                                              ▼                   ▼
                                                         p(s | x, t)         p(t | x, c)
                                                              └─────────┬─────────┘
                                                                        ▼
                                                  marginalization  p(s | x, c) = Σ_t p_θ·p_φ  (Eq. 2)
                                                                        ▼
                                         train: NLL (Eq. 5)   |   infer: E[s|x,c] = Σ_t s̄·p_φ (Eq. 6)
```

---

## Equations ↔ code

| Paper | What | File |
|---|---|---|
| Eq. 1 | CLIP-IQA score `s̄(x,t)` (parameter-free) | `score_model.py` |
| Eq. 3 | Prompt selection `p_φ(t\|x,c)` | `selection_model.py` |
| Eq. 4 | Gaussian score model `p_θ(s\|x,t)` | `model.py`, `config.py` |
| Eq. 2 | Prompt marginalization `p(s\|x,c)` | `model.py` |
| Eq. 5 | Gaussian-mixture NLL objective | `model.py: nll_loss` |
| Eq. 6 | Expected-score inference | `model.py: predict` |

---

## Modules

- **`config.py`** — all hyperparameters in frozen dataclasses, each cited to the paper.
  `resolve_device()` selects `mps → cuda → cpu`.
- **`prompts.py`** — the 260 antonym pairs, transcribed verbatim from supplementary tables
  S6–S9 (44 aesthetic-attribute + 36 quality-attribute + 100 general-aesthetic + 80
  general-quality). Generated file; do not hand-edit.
- **`clip_backbone.py`** — OpenCLIP ViT-B/16. Text encoder fully frozen; only the last 4
  image-encoder blocks are trainable (these are `theta`). `encode_image` / `encode_text`
  return L2-normalized embeddings (so inner product = cosine).
- **`score_model.py`** — `clipiqa_score`: `s̄ = sigmoid(I·T_pos − I·T_neg)`, the two-way
  softmax of Eq. 1. No parameters of its own.
- **`selection_model.py`** — `PromptSelectionModel`: two 2-layer MLPs map `[I(x); e_c]` and
  `[T(t_pos)−T(t_neg); e_c]` into a shared space (256→128); softmax of their inner products
  gives `p_φ`. `e_c` is a learnable task embedding (`C=4`).
- **`model.py`** — `PPA`: precomputes frozen prompt embeddings, marginalizes, and exposes
  `predict` (Eq. 6) and `nll_loss` (Eq. 5, logsumexp for stability). Implements
  prompt-dropout (below).
- **`data.py`** — `ParaGiaaDataset` over the official PARA GIAA split; target is the mean
  aesthetic score mapped to `[0, 1]` via `(s−1)/4`.
- **`train.py`** — Adam loop over `theta + phi`, SRCC/PLCC evaluation (no scipy dependency),
  best/last checkpointing.

### What is trained vs frozen

| Part | Trained? | Symbol | Params |
|---|---|---|---|
| CLIP text encoder | frozen | — | 0 |
| CLIP image encoder (early) | frozen | — | 0 |
| CLIP image encoder, last 4 blocks | ✅ | `theta` | ~28.4M |
| Prompt selection MLPs + task embedding | ✅ | `phi` | ~0.33M |
| **Total trainable** | | | **28,681,988** |

The score `s̄` has no parameters; it is learned indirectly because `theta` moves the image
embedding `I(x)`.

---

## Hyperparameters (from the paper)

| Item | Value | Source |
|---|---|---|
| Backbone | CLIP ViT-B/16 | Sec. 4.2 |
| Trainable image blocks | last 4 | Sec. 4.2 |
| Task embedding dim `C` | 4 | Sec. 4.2 + supp. S4 |
| MLP hidden / output | 256 / 128 | Sec. 4.2 |
| `sigma` (Gaussian) | 0.1 | Sec. 4.2 + supp. S5 |
| Optimizer | Adam, lr 1e-5, β (0.9, 0.999) | Sec. 4.2 |
| Prompt pool | 260 antonym pairs | Sec. 3.4 + supp. S6–S9 |
| Prompt-dropout | 50% of prompts disabled per iteration | supp. Sec. B |

### Decisions where the paper is silent

- **No logit scale on Eq. 1.** Neither the paper nor the supplementary applies a CLIP
  temperature to the score; the faithful run follows the raw inner product. CLIP-IQA
  conventionally uses one, and adding it back (as a learnable `tau`, `--learnable-temp`)
  closes the PLCC gap — see Results. The base stays paper-literal (off) by default.
- **Score normalization** `(s−1)/4`. The paper does not specify how raw dataset labels map to
  `s̄ ∈ (0,1)`; this is our choice. It does **not** affect SRCC/PLCC (both are invariant to a
  linear rescaling of the targets), so it is not the cause of the PLCC gap.
- **Epochs / batch size / image resolution / weight decay / scheduler / augmentation** are
  not specified; we use sensible defaults (see `config.py`).
- **Single-task PARA.** The paper trains one unified model over 12 datasets; we reproduce
  PARA alone (`n_tasks=1`).

---

## Dataset

PARA (Yang et al., *Personalized Image Aesthetics Assessment with Rich Attributes*, CVPR
2022). We use the official GIAA split (`PARA-GiaaTrain.csv` / `PARA-GiaaTest.csv`, one row
per image; 28,220 / 3,000). Place it at `data/PARA/` with `imgs/` and `annotation/`.

---

## Results (PARA, GIAA)

PARA publishes only a train and a test GIAA split, so a 10% validation split is carved out of
the training set. The checkpoint **and** the reported configuration are both selected on
validation; the test split is evaluated exactly once per configuration, after training ends.
The split, the protocol and the per-epoch curves are in
[`docs/para_val_reeval.md`](../../docs/para_val_reeval.md).

| Model | SRCC | PLCC |
|---|---|---|
| PPA (paper, Table 4) | 0.913 | 0.942 |
| PPA-T (paper, Table 4) | 0.919 | 0.947 |
| GIAA (PARA paper) | 0.902 | 0.936 |
| This reproduction, paper-literal | 0.9160 | 0.9073 |
| **This reproduction, `+ learnable temperature`** | **0.9164** | **0.9456** |

The paper-literal run (no temperature on Eq. 1) matches the paper on SRCC but is ~0.035 low on
PLCC. The cause is **not** label normalization — PLCC (Pearson) is invariant to any linear
rescaling of the targets, so the `(s-1)/4` mapping cannot move it. The cause is the score
`sigmoid`: with a unit logit, predictions are squashed into a narrow band (std ~ 0.085 vs.
target std ~ 0.133), so the value scale, not the ranking, suffers.

Adding a single learnable temperature `tau` to the score logit
(`score = tau * (sim_pos - sim_neg)`, `--learnable-temp`) widens the predictions and lifts PLCC
to 0.9456, with SRCC unchanged. `tau` converges to ~10 from an init of 10 (unit scale is too
flat; the raw CLIP scale ~100 saturates the sigmoid). This temperature is an addition the paper
does not describe, so it is **off by default** to keep the base faithful.

Validation SRCC is flat from epoch 2 onward: across epochs 2-20 the paper-literal curve moves
within a 0.0062 band, and the selected epoch beats epoch 3 by 0.0008. The selected epoch number
therefore carries essentially no information — validation-based selection is what keeps the test
split honest, not what buys accuracy here.

---

## Run

```bash
uv sync                                             # install
uv run python scripts/build_para_val_split.py       # verify the committed validation split
PYTHONPATH=src uv run python scripts/smoke_test.py  # shape check, no data needed

# paper-literal
PYTHONPATH=src uv run python scripts/train.py \
  --ckpt-dir checkpoints/para/giaa
# + learnable score temperature
PYTHONPATH=src uv run python scripts/train.py --learnable-temp \
  --ckpt-dir checkpoints/para/learnable_temp
```

Other flags: `--epochs N`, `--batch-size N`, `--device {cuda,mps,cpu}`.

Each run writes `best.pt` (highest **validation** SRCC), `last.pt` and `final_metrics.json`
(selected epoch, its validation metrics, and the single test evaluation) into the checkpoint
directory. Checkpoints are git-ignored.

---

## Known deviations / TODO

- [x] Close the PLCC gap — done via a learnable score temperature (`--learnable-temp`).
      See Results.
- [ ] QuickGELU mismatch warning from OpenCLIP (`openai` weights expect QuickGELU); consider
      `ViT-B-16-quickgelu`. Minor accuracy effect.
- [ ] Multi-dataset unified training (paper's full setting) is out of scope for this single
      PARA reproduction.
