# ppa-reimplementation

An unofficial, from-scratch reimplementation of **PPA** (Probabilistic Prompt Adaptation),
reproduced on PARA.

> Takayuki Hara and Yuya Otsuka. *Probabilistic Prompt Adaptation for Unified Image Aesthetics
> and Quality Assessment.* CVPR 2026.
> [paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Hara_Probabilistic_Prompt_Adaptation_for_Unified_Image_Aesthetics_and_Quality_Assessment_CVPR_2026_paper.pdf)

No official implementation has been released, so everything here is written from the published
paper and its supplement. This repository is **not affiliated with, nor endorsed by, the
original authors or their institution.** Any deviation from the paper is ours.

## What PPA does

An image is scored along a fixed pool of 260 human-readable antonym prompt pairs
(`"Sharp photo." / "Blurry photo."`, …). Two paths run over that pool:

- **Score** `p_θ(s | x, t)` — CLIP-IQA on one prompt pair: how sharp / how colourful this image is.
- **Selection** `p_φ(t | x, c)` — how much that pair should count for this image and task.

The prediction marginalizes the first over the second, `p(s | x, c) = Σ_t p_θ(s | x, t) p_φ(t | x, c)`,
and the model is fit by the NLL of that mixture. Because the pool is readable, the weights say
which criteria drove a score.

Architecture notes with the paper's equation numbers, the dimensions, and the places where the
paper is silent are in [`src/ppa/README.md`](src/ppa/README.md).

## Reproduction result (PARA, GIAA)

| Model | SRCC | PLCC |
|---|---|---|
| PPA (paper, Table 4) | 0.913 | 0.942 |
| This reproduction, paper-literal | 0.9160 | 0.9073 |
| **This reproduction, `+ learnable temperature`** | **0.9164** | **0.9456** |

PARA publishes only a train and a test GIAA split. A 10% validation split is carved out of the
training set, and **both** the checkpoint and the reported configuration are selected on it; the
test split is evaluated exactly once per configuration, after training. The protocol, the split,
the per-epoch curves and what the numbers do and do not support are in
[`docs/para_val_reeval.md`](docs/para_val_reeval.md).

The learnable score temperature is an addition the paper does not describe. It is off by default
so the base stays faithful; see [`src/ppa/README.md`](src/ppa/README.md) for why the paper-literal
run is low on PLCC and what the temperature fixes.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

PARA (Yang et al., CVPR 2022) is not redistributed here. Obtain it from the authors and place it
at `data/PARA/` with `imgs/` and `annotation/` (the official `PARA-GiaaTrain.csv` /
`PARA-GiaaTest.csv` are used as-is).

```bash
uv run python scripts/build_para_val_split.py       # verify the committed validation split
PYTHONPATH=src uv run python scripts/smoke_test.py  # shape check, runs without the dataset

PYTHONPATH=src uv run python scripts/train.py --ckpt-dir checkpoints/para/giaa
PYTHONPATH=src uv run python scripts/train.py --learnable-temp \
  --ckpt-dir checkpoints/para/learnable_temp
```

## Layout

```
src/ppa/            model (backbone, score path, selection path, marginalization, NLL)
scripts/            training entry point, smoke test, validation split builder
splits/             the committed validation split membership
docs/               re-evaluation protocol and results
```

## Citation

This is a reimplementation; cite the original paper.

```bibtex
@inproceedings{Hara_2026_CVPR,
  author    = {Hara, Takayuki and Otsuka, Yuya},
  title     = {Probabilistic Prompt Adaptation for Unified Image Aesthetics and Quality Assessment},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2026},
  pages     = {37236-37246},
}
```

## License

See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
