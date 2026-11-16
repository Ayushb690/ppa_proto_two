"""Build and/or validate the PARA GIAA train/validation split.

PARA ships only two GIAA splits (``PARA-GiaaTrain.csv`` / ``PARA-GiaaTest.csv``),
so a run that selects its checkpoint on the test split leaks the evaluation set
into model selection. This script carves a validation set out of the *training*
split so the test split is untouched until the final report.

Protocol:
- one row per image (``imageName`` is unique within PARA-GiaaTrain.csv), so an
  image-level split is simply a row-level split; train and val are disjoint.
- stratified on ``aestheticScore_mean`` over 7 equal-width bins spanning the
  PARA score scale [1, 5]. The scale, not the observed range, defines the bin
  edges so the binning does not shift if the CSV changes; the top bin is empty
  in practice (max observed mean is 4.42) and is simply skipped.
- 10% of each bin goes to validation, drawn with a seeded ``numpy`` Generator.

The resulting validation membership is written to ``splits/para_giaa_val_seed42.csv``
and committed, so the split is reproducible even if this script later changes.
The train side is the complement (PARA-GiaaTrain.csv minus these rows).

Modes:
- default (``--validate``): regenerate in memory and assert it matches the file
  on disk, plus the disjointness/coverage invariants. Idempotent, non-mutating.
- ``--rebuild --force``: regenerate and overwrite the file. Use only when the
  split protocol itself changes; it invalidates every result selected on it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = REPO_ROOT / "data" / "PARA" / "annotation" / "PARA-GiaaTrain.csv"
SPLIT_CSV = REPO_ROOT / "splits" / "para_giaa_val_seed42.csv"

TARGET_COL = "aestheticScore_mean"
SCORE_RANGE = (1.0, 5.0)
N_BINS = 7
VAL_FRAC = 0.1
SEED = 42


def load_train_table() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_CSV, usecols=["imageName", "sessionId", TARGET_COL])
    if df["imageName"].duplicated().any():
        raise ValueError("imageName is not unique in PARA-GiaaTrain.csv; split assumption broken")
    return df


def assign_bins(scores: pd.Series) -> np.ndarray:
    edges = np.linspace(*SCORE_RANGE, N_BINS + 1)
    # Clip so out-of-scale scores (none today) land in the edge bins instead of NaN.
    return np.clip(np.digitize(scores.to_numpy(), edges[1:-1]), 0, N_BINS - 1)


def build_val_table(df: pd.DataFrame) -> pd.DataFrame:
    bins = assign_bins(df[TARGET_COL])
    rng = np.random.default_rng(SEED)
    picked: list[np.ndarray] = []
    for b in range(N_BINS):
        idx = np.flatnonzero(bins == b)
        if idx.size == 0:
            continue
        n_val = int(round(idx.size * VAL_FRAC))
        if n_val == 0:
            continue
        picked.append(rng.choice(idx, size=n_val, replace=False))
    val_idx = np.sort(np.concatenate(picked))
    val = df.iloc[val_idx].copy()
    val["bin"] = bins[val_idx]
    return val.sort_values("imageName").reset_index(drop=True)


def _check_invariants(df: pd.DataFrame, val: pd.DataFrame) -> None:
    val_names = set(val["imageName"])
    all_names = set(df["imageName"])
    assert len(val_names) == len(val), "duplicate imageName inside the validation split"
    assert val_names <= all_names, "validation contains images absent from PARA-GiaaTrain.csv"
    train_names = all_names - val_names
    assert not (train_names & val_names), "train/val overlap"
    assert len(train_names) + len(val_names) == len(all_names), "train/val do not cover the split"


def _report(df: pd.DataFrame, val: pd.DataFrame) -> None:
    bins = assign_bins(df[TARGET_COL])
    edges = np.linspace(*SCORE_RANGE, N_BINS + 1)
    print(f"train rows      : {len(df)}")
    print(f"validation rows : {len(val)}  ({len(val) / len(df):.3%})")
    print(f"train after cut : {len(df) - len(val)}")
    print("bin  range              all     val")
    for b in range(N_BINS):
        n_all = int((bins == b).sum())
        n_val = int((val['bin'] == b).sum())
        print(f"  {b}  [{edges[b]:.3f}, {edges[b + 1]:.3f})  {n_all:6d}  {n_val:6d}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rebuild", action="store_true", help="regenerate the split file")
    p.add_argument("--force", action="store_true", help="required together with --rebuild")
    args = p.parse_args()

    df = load_train_table()
    val = build_val_table(df)
    _check_invariants(df, val)

    if args.rebuild:
        if not args.force:
            sys.exit("--rebuild overwrites an existing split; pass --force to confirm")
        SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
        val.to_csv(SPLIT_CSV, index=False)
        print(f"wrote {SPLIT_CSV.relative_to(REPO_ROOT)}")
        _report(df, val)
        return

    if not SPLIT_CSV.exists():
        sys.exit(f"{SPLIT_CSV.relative_to(REPO_ROOT)} is missing; run with --rebuild --force")
    on_disk = pd.read_csv(SPLIT_CSV)
    if list(on_disk["imageName"]) != list(val["imageName"]):
        sys.exit("split file on disk does not match the regenerated split")
    print(f"{SPLIT_CSV.relative_to(REPO_ROOT)} matches the regenerated split")
    _report(df, val)


if __name__ == "__main__":
    main()
