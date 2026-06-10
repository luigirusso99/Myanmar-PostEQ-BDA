from pathlib import Path
from typing import Optional, List, Tuple, Dict
import json

import cv2
import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold


def _resolve_path(root: Path, sample_id: str, suffix: str) -> Path:
    path = root / f"{sample_id}_{suffix}.tif"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


class SarRgbFootprintDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        ids: Optional[List[str]] = None,
        manifest_name: str = "patch_list.csv",
        cache: bool = False,
        return_id: bool = True,
        image_size: int = 224,
    ):
        self.root = Path(root_dir)
        self.image_size = int(image_size)
        df = pd.read_csv(self.root / manifest_name)
        if not {"id", "label"}.issubset(df.columns):
            raise ValueError("Manifest must contain columns: id,label")

        if ids is not None:
            ids = [str(x) for x in ids]
            df = df[df["id"].astype(str).isin(ids)].copy()

        self.records = []
        for _, row in df.iterrows():
            sid = str(row["id"])
            try:
                self.records.append((
                    sid,
                    int(row["label"]),
                    _resolve_path(self.root, sid, "SAR"),
                    _resolve_path(self.root, sid, "RGB"),
                    _resolve_path(self.root, sid, "SARftp"),
                ))
            except FileNotFoundError:
                continue

        if not self.records:
            raise RuntimeError("No complete SAR/RGB/footprint samples found.")

        self.ids = [r[0] for r in self.records]
        self.labels = [r[1] for r in self.records]
        self.cache = cache
        self.return_id = return_id
        self._buf: Dict[str, Dict[str, np.ndarray]] = {}

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _percentile_stretch(arr: np.ndarray) -> np.ndarray:
        for i in range(arr.shape[0]):
            p1, p99 = np.percentile(arr[i], (1, 99))
            arr[i] = np.clip(arr[i], p1, p99)
            arr[i] = (arr[i] - p1) / (p99 - p1 + 1e-6)
        return arr

    def _resize_channels(self, arr: np.ndarray, interpolation) -> np.ndarray:
        out = np.zeros((arr.shape[0], self.image_size, self.image_size), dtype=np.float32)
        for i in range(arr.shape[0]):
            out[i] = cv2.resize(arr[i], (self.image_size, self.image_size), interpolation=interpolation)
        return out

    def _read_sar(self, path: Path, add_ratio: bool = True) -> np.ndarray:
        with rasterio.open(path) as ds:
            arr = ds.read([1, 2]).astype(np.float32)
        arr = self._resize_channels(arr, cv2.INTER_LINEAR_EXACT)
        band_1, band_2 = arr[0], arr[1]
        ratio = np.divide(band_1, band_2, out=np.zeros_like(band_1), where=np.abs(band_2) > 1e-6)
        sar = np.stack([band_1, band_2, ratio if add_ratio else band_2], axis=0)
        return self._percentile_stretch(sar)

    def _read_rgb(self, path: Path) -> np.ndarray:
        with rasterio.open(path) as ds:
            arr = ds.read([1, 2, 3]).astype(np.float32)
        arr = self._resize_channels(arr, cv2.INTER_LINEAR_EXACT)
        return self._percentile_stretch(arr)

    def _read_footprint(self, path: Path) -> np.ndarray:
        with rasterio.open(path) as ds:
            gt = ds.read(1).astype(np.float32)
        gt_bin = (gt != 0).astype(np.float32)
        gt_resized = cv2.resize(gt_bin, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return np.stack([gt_resized] * 3, axis=0)

    def __getitem__(self, idx: int):
        sid, label, sar_path, rgb_path, ftp_path = self.records[idx]
        if self.cache and sid in self._buf:
            data = self._buf[sid]
        else:
            data = {
                "sar": self._read_sar(sar_path),
                "rgb": self._read_rgb(rgb_path),
                "ftp": self._read_footprint(ftp_path),
            }
            if self.cache:
                self._buf[sid] = data

        x = (
            torch.from_numpy(data["sar"]).float(),
            torch.from_numpy(data["rgb"]).float(),
            torch.from_numpy(data["ftp"]).float(),
        )
        y = torch.tensor(label, dtype=torch.long)
        return (x, y, sid) if self.return_id else (x, y)



def filter_invalid_samples(root_dir: str, manifest_name: str = "patch_list.csv") -> pd.DataFrame:
    root = Path(root_dir)
    df = pd.read_csv(root / manifest_name)
    keep = []
    removed = 0
    for _, row in df.iterrows():
        sid = str(row["id"])
        try:
            with rasterio.open(root / f"{sid}_RGB.tif") as ds:
                rgb = ds.read()
            with rasterio.open(root / f"{sid}_SAR.tif") as ds:
                sar = ds.read([1, 2])
            with rasterio.open(root / f"{sid}_SARftp.tif") as ds:
                ftp = ds.read(1)

            valid_rgb = np.isfinite(rgb).all() and np.any(rgb != 0)
            valid_sar = np.isfinite(sar).all() and np.any(sar != 0)
            valid_ftp = np.isfinite(ftp).all() and np.any(ftp != 0)

            if valid_rgb and valid_sar and valid_ftp:
                keep.append((sid, int(row["label"])))
            else:
                removed += 1
        except Exception:
            removed += 1
    out = pd.DataFrame(keep, columns=["id", "label"])
    print(f"Removed {removed} samples with missing/zero SAR, RGB, or footprint. Kept {len(out)}.")
    return out


def filter_zero_rgb_samples(root_dir: str, manifest_name: str = "patch_list.csv") -> pd.DataFrame:
    return filter_invalid_samples(root_dir, manifest_name)


def make_stratified_folds(root_dir: str, out_dir: Optional[str] = None, n_splits: int = 5, seed: int = 42):
    root = Path(root_dir)
    folds_dir = Path(out_dir) if out_dir else root / f"folds_seed{seed}"
    folds_dir.mkdir(parents=True, exist_ok=True)

    df = filter_invalid_samples(root_dir)
    ids = df["id"].astype(str).tolist()
    y = df["label"].astype(int).to_numpy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    summary = []
    for k, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(ids)), y)):
        train_ids = [ids[i] for i in tr_idx]
        val_ids = [ids[i] for i in va_idx]
        folds.append((train_ids, val_ids))

        df[df["id"].astype(str).isin(train_ids)].to_csv(folds_dir / f"fold_{k}_train.csv", index=False)
        df[df["id"].astype(str).isin(val_ids)].to_csv(folds_dir / f"fold_{k}_val.csv", index=False)
        summary.append({
            "fold": k,
            "train_total": len(train_ids),
            "train_pos": int(df[df["id"].astype(str).isin(train_ids)]["label"].sum()),
            "val_total": len(val_ids),
            "val_pos": int(df[df["id"].astype(str).isin(val_ids)]["label"].sum()),
        })

    pd.DataFrame(summary).to_csv(folds_dir / "summary.csv", index=False)
    (folds_dir / "meta.json").write_text(json.dumps({"root_dir": str(root), "seed": seed, "n_splits": n_splits}, indent=2))
    return folds


def make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool, sampler=None, drop_last: bool = False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )

if __name__ == '__main__':
    from pathlib import Path

    from src.bdd.dataset import (
        SarRgbFootprintDataset,
        make_stratified_folds,
        make_loader,
    )


    ROOT_DIR = "/home/silvia/Desktop/GIGI/ASI_WGD_2026_Myanmar/BDD/data/patches"
    SEED = 42
    N_SPLITS = 5


    def main():
        folds = make_stratified_folds(
            root_dir=ROOT_DIR,
            n_splits=N_SPLITS,
            seed=SEED,
        )

        train_ids, val_ids = folds[0]

        train_ds = SarRgbFootprintDataset(
            root_dir=ROOT_DIR,
            ids=train_ids,
            image_size=224,
            return_id=True,
        )

        val_ds = SarRgbFootprintDataset(
            root_dir=ROOT_DIR,
            ids=val_ids,
            image_size=224,
            return_id=True,
        )

        train_loader = make_loader(
            train_ds,
            batch_size=8,
            num_workers=0,
            shuffle=True,
        )

        (sar, rgb, ftp), y, sid = next(iter(train_loader))

        print("Train samples:", len(train_ds))
        print("Val samples:", len(val_ds))
        print("SAR shape:", sar.shape)
        print("RGB shape:", rgb.shape)
        print("FTP shape:", ftp.shape)
        print("Label shape:", y.shape)
        print("Example IDs:", sid[:3])