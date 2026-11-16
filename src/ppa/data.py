"""PARA dataset loader for the GIAA (generic) reproduction setting.

PARA ships pre-aggregated GIAA splits (PARA-GiaaTrain.csv / PARA-GiaaTest.csv),
one row per image. The supervision target is the per-image mean aesthetic score
(``aestheticScore_mean``, in [1, 5]), mapped to [0, 1] via (s - 1) / 4 to match
the CLIP-IQA score range s_bar in (0, 1).

PARA provides no validation split, so ``scripts/build_para_val_split.py`` carves
one out of PARA-GiaaTrain.csv (10%, stratified on the mean score, seed 42) and
records it in ``splits/para_giaa_val_seed42.csv``. Training uses the remaining
90%, checkpoint selection uses the validation split, and PARA-GiaaTest.csv is
touched only for the final report.

Each dataset corresponds to one task c (Sec. 3.2); for single-dataset PARA the
task id is constant (0).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import REPO_ROOT, Config

_TARGET_COL = "aestheticScore_mean"
_TASK_ID = 0  # single-task GIAA reproduction
_VAL_SPLIT_CSV = REPO_ROOT / "splits" / "para_giaa_val_seed42.csv"


class ParaGiaaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_root: Path, preprocess, cfg: Config):
        self.img_root = img_root
        self.preprocess = preprocess
        self.cfg = cfg
        self.rows = df[["imageName", "sessionId", _TARGET_COL]].to_records(index=False)

    @classmethod
    def from_csv(cls, csv_path: Path, img_root: Path, preprocess, cfg: Config) -> "ParaGiaaDataset":
        df = pd.read_csv(csv_path, usecols=["imageName", "sessionId", _TARGET_COL])
        return cls(df, img_root, preprocess, cfg)

    def __len__(self) -> int:
        return len(self.rows)

    def _normalize(self, score: float) -> float:
        lo, hi = self.cfg.train.score_min, self.cfg.train.score_max
        return (score - lo) / (hi - lo)

    def __getitem__(self, i: int):
        name, session, mean = self.rows[i]
        path = self.img_root / session / name
        image = self.preprocess(Image.open(path).convert("RGB"))
        target = torch.tensor(self._normalize(float(mean)), dtype=torch.float32)
        task_id = torch.tensor(_TASK_ID, dtype=torch.long)
        return image, task_id, target


def split_train_val(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split PARA-GiaaTrain.csv rows by the committed validation image list."""
    if not _VAL_SPLIT_CSV.exists():
        raise FileNotFoundError(
            f"{_VAL_SPLIT_CSV} is missing; "
            "run `uv run python scripts/build_para_val_split.py --rebuild --force`"
        )
    val_names = set(pd.read_csv(_VAL_SPLIT_CSV, usecols=["imageName"])["imageName"])
    is_val = train_df["imageName"].isin(val_names)
    if int(is_val.sum()) != len(val_names):
        raise ValueError("validation split references images absent from PARA-GiaaTrain.csv")
    return train_df[~is_val], train_df[is_val]


def _loader(dataset: Dataset, cfg: Config, *, shuffle: bool, drop_last: bool = False) -> DataLoader:
    return DataLoader(
        dataset, batch_size=cfg.train.batch_size, shuffle=shuffle,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.device == "cuda",  # pin_memory only helps CUDA host->device copies
        drop_last=drop_last,
    )


def build_dataloaders(cfg: Config, preprocess) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader) for the PARA GIAA split."""
    ann = cfg.data_root / "annotation"
    img_root = cfg.data_root / "imgs"

    full_train = pd.read_csv(
        ann / "PARA-GiaaTrain.csv", usecols=["imageName", "sessionId", _TARGET_COL]
    )
    train_df, val_df = split_train_val(full_train)

    # Use only 5,000 training images
   # TEMPORARY: quick training sanity check
    train_df = train_df.sample(
    n=min(128, len(train_df)),
    random_state=42
    ).reset_index(drop=True)

    val_df = val_df.sample(
    n=min(64, len(val_df)),
    random_state=42
    ).reset_index(drop=True)

    train_ds = ParaGiaaDataset(train_df, img_root, preprocess, cfg)
    val_ds = ParaGiaaDataset(val_df, img_root, preprocess, cfg)
    test_ds = ParaGiaaDataset.from_csv(ann / "PARA-GiaaTest.csv", img_root, preprocess, cfg)

    return (
        _loader(train_ds, cfg, shuffle=True, drop_last=True),
        _loader(val_ds, cfg, shuffle=False),
        _loader(test_ds, cfg, shuffle=False),
    )
