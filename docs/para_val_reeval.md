# PARA re-evaluation with a held-out validation split

PARA ships only two GIAA splits (`PARA-GiaaTrain.csv` / `PARA-GiaaTest.csv`). The earlier
reproduction therefore selected `best.pt` by **test** SRCC, which leaks the evaluation set into
model selection. This document reports the same two configurations re-trained with a proper
validation split: the checkpoint *and* the winning configuration are chosen on validation, and
the test split is evaluated exactly once per configuration, after training ends.

## 1. The validation split

Built by `scripts/build_para_val_split.py`, membership committed at
`splits/para_giaa_val_seed42.csv`.

- **Source**: `PARA-GiaaTrain.csv`, 28,220 rows. `imageName` is unique, so a row-level split is
  an image-level split; train and validation are disjoint by construction.
- **Stratification**: 7 equal-width bins over the PARA score scale `[1, 5]` on
  `aestheticScore_mean`. The bin edges come from the scale, not the observed range, so they do
  not shift if the CSV changes. The top bin `[4.429, 5.000)` is empty (max observed mean is 4.42)
  and is skipped.
- **Fraction**: 10% of each non-empty bin.
- **Seed**: 42 (`numpy.random.default_rng(42)`).
- **Result**: 2,822 validation images (10.000%), 25,398 training images.

| bin | range | all | val |
|---|---|---|---|
| 0 | [1.000, 1.571) | 455 | 46 |
| 1 | [1.571, 2.143) | 1,400 | 140 |
| 2 | [2.143, 2.714) | 2,700 | 270 |
| 3 | [2.714, 3.286) | 11,192 | 1,119 |
| 4 | [3.286, 3.857) | 10,779 | 1,078 |
| 5 | [3.857, 4.429) | 1,694 | 169 |
| 6 | [4.429, 5.000) | 0 | 0 |

Verified disjoint: `train ∩ val = 0`, `train ∩ test = 0`, `val ∩ test = 0`.
Running the script with no arguments re-derives the split and asserts it matches the committed
file, so the split is reproducible without trusting the generator to stay unchanged.

## 2. Results

Both configurations: 20 epochs, Adam lr 1e-5, batch 32, seed 42, identical to the earlier runs
except for the split. Checkpoint selected by **validation SRCC**.

| Configuration | selected epoch | val SRCC | val PLCC | test SRCC | test PLCC |
|---|---|---|---|---|---|
| paper-literal | 18 | 0.9191 | 0.9124 | 0.9160 | 0.9073 |
| **`--learnable-temp`** (selected) | **19** | **0.9217** | **0.9502** | **0.9164** | **0.9456** |
| PPA paper (Hara & Otsuka, CVPR 2026) | — | — | — | 0.913 | 0.942 |

**`--learnable-temp` is the headline configuration**, chosen because it wins on **validation**
SRCC (0.9217 vs 0.9191) and validation PLCC (0.9502 vs 0.9124). The test columns are reported
for both configurations as reference values only; they played no part in either the checkpoint
choice or the configuration choice. Deciding the winner on test would have moved the leak we
just removed from the checkpoint level up to the configuration level.

On test, the selected configuration exceeds the paper on both metrics (SRCC 0.9164 vs 0.913,
PLCC 0.9456 vs 0.942), so the reproduction is successful. The paper-literal configuration matches
the paper on SRCC but sits ~0.035 below on PLCC — the gap the learnable score temperature closes.

### Comparison with the earlier numbers

`src/ppa/README.md` reports 0.917 SRCC / 0.944 PLCC for `--learnable-temp`. **Those numbers are
not directly comparable to the table above**: they were produced by selecting `best.pt` on the
test split, and they were trained on all 28,220 training images rather than 25,398. They carry
an optimistic bias of unknown size and should be read as an upper bound, not as a measurement.
The numbers in this document supersede them for any comparison against the paper.

## 3. The validation curve is flat — do not read into the selected epoch

The selected epoch numbers (18 and 19) carry essentially no information.

For paper-literal, the selected epoch 18 (val SRCC 0.9191) beats epoch 3 (0.9183) by **0.0008**.
Across epochs 2–20 the entire validation SRCC curve moves within a 0.0062 band. A 0.0008
difference between two candidate epochs is inside that band; it is curve noise, not a signal
that epoch 18 learned something epoch 3 had not. We trained one seed per configuration, so this
document makes no claim about run-to-run variance — but a gap that small would not survive a
different seed.

`--learnable-temp` behaves the same way: after epoch 2 its validation SRCC stays within a 0.0068
band (0.9149–0.9217) and its validation PLCC within 0.0040 (0.9462–0.9502). Its selected epoch 19
(0.921707) edges out epoch 3 by less than 0.0001 — both round to 0.9217. Meanwhile its training NLL collapses
from 0.202 to 0.005 — the model fits the training set far harder over those 17 epochs and the
validation metrics do not move.

This contradicts the claim in `src/ppa/README.md` that "training shows mild overfitting after ~3
epochs (train NLL keeps dropping, test SRCC plateaus), so `best.pt` is selected by test SRCC".
The plateau is real; the implied benefit of picking the peak is not. The accurate statement is:
**validation is flat from epoch 2 onward, and any epoch from 2 to 20 yields essentially the same
SRCC.** Checkpoint selection on validation is still the right protocol — it is what keeps the
test split honest — but it is not buying accuracy here.

## 4. Per-epoch validation curves

Extracted from the training logs (`[epoch N]` lines only).

| epoch | paper-literal train NLL | val SRCC | val PLCC | | `--learnable-temp` train NLL | val SRCC | val PLCC |
|---|---|---|---|---|---|---|---|
| 1 | 0.6173 | 0.9028 | 0.8920 | | 0.2016 | 0.9175 | 0.9446 |
| 2 | 0.2643 | 0.9163 | 0.8988 | | 0.0915 | 0.9214 | 0.9500 |
| 3 | 0.2356 | 0.9183 | 0.9046 | | 0.0640 | 0.9217 | 0.9490 |
| 4 | 0.2136 | 0.9161 | 0.9038 | | 0.0416 | 0.9181 | 0.9480 |
| 5 | 0.2076 | 0.9154 | 0.9100 | | 0.0249 | 0.9178 | 0.9466 |
| 6 | 0.1914 | 0.9129 | 0.9016 | | 0.0155 | 0.9162 | 0.9466 |
| 7 | 0.1939 | 0.9154 | 0.9128 | | 0.0114 | 0.9149 | 0.9462 |
| 8 | 0.1823 | 0.9137 | 0.9127 | | 0.0095 | 0.9163 | 0.9470 |
| 9 | 0.1867 | 0.9156 | 0.9095 | | 0.0091 | 0.9183 | 0.9484 |
| 10 | 0.1857 | 0.9156 | 0.9088 | | 0.0082 | 0.9182 | 0.9479 |
| 11 | 0.1790 | 0.9155 | 0.9101 | | 0.0074 | 0.9171 | 0.9473 |
| 12 | 0.1809 | 0.9170 | 0.9143 | | 0.0074 | 0.9202 | 0.9492 |
| 13 | 0.1868 | 0.9171 | 0.9126 | | 0.0069 | 0.9179 | 0.9485 |
| 14 | 0.1868 | 0.9177 | 0.9118 | | 0.0062 | 0.9201 | 0.9491 |
| 15 | 0.1801 | 0.9178 | 0.9133 | | 0.0057 | 0.9204 | 0.9496 |
| 16 | 0.1778 | 0.9178 | 0.9127 | | 0.0055 | 0.9201 | 0.9488 |
| 17 | 0.1802 | 0.9176 | 0.9132 | | 0.0060 | 0.9210 | 0.9498 |
| **18** | 0.1853 | **0.9191** | 0.9124 | | 0.0057 | 0.9216 | 0.9498 |
| **19** | 0.1847 | 0.9189 | 0.9112 | | 0.0050 | **0.9217** | 0.9502 |
| 20 | 0.1814 | 0.9184 | 0.9126 | | 0.0048 | 0.9214 | 0.9501 |

Bold marks the epoch selected by validation SRCC for each configuration.

## 5. Reproducing

```sh
uv run python scripts/build_para_val_split.py                     # verify the committed split
uv run python scripts/build_para_val_split.py --rebuild --force   # regenerate it

PYTHONPATH=src uv run python scripts/train.py \
  --ckpt-dir checkpoints/para/val_split/giaa
PYTHONPATH=src uv run python scripts/train.py --learnable-temp \
  --ckpt-dir checkpoints/para/val_split/learnable_temp
```

Each run writes `best.pt` (highest validation SRCC), `last.pt`, and `final_metrics.json`
(selected epoch, its validation metrics, and the single test evaluation) into the checkpoint
directory.
