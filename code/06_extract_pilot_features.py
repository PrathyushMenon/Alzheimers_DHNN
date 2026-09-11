"""Extract compact pilot features from MRI, PET SUVR, and DWI NIfTI files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom


LABELS = {"CN": 0, "MCI": 1, "AD": 2}


def robust_norm(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    lo, hi = np.percentile(finite, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(data, dtype=np.float32)
    data = np.clip(data, lo, hi)
    return ((data - lo) / (hi - lo)).astype(np.float32)


def resize3d(data: np.ndarray, target: tuple[int, int, int]) -> np.ndarray:
    factors = [t / s for t, s in zip(target, data.shape[:3])]
    return zoom(data, factors, order=1).astype(np.float32)


def stats(data: np.ndarray, prefix: str) -> dict[str, float]:
    arr = np.asarray(data, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        arr = np.array([0.0], dtype=np.float32)
    qs = np.percentile(arr, [1, 5, 25, 50, 75, 95, 99])
    out = {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
    }
    for q, v in zip([1, 5, 25, 50, 75, 95, 99], qs):
        out[f"{prefix}_p{q}"] = float(v)
    return out


def load_nii(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(str(path)).dataobj, dtype=np.float32)


def modality_features(path: Path, target: tuple[int, int, int], prefix: str) -> tuple[np.ndarray, dict[str, float]]:
    data = load_nii(path)
    if data.ndim == 4:
        mean_vol = np.mean(data, axis=3)
        first_vol = data[..., 0]
        st = stats(mean_vol, prefix + "_meanvol")
        st.update(stats(first_vol, prefix + "_firstvol"))
        vol = mean_vol
    else:
        st = stats(data, prefix)
        vol = data
    vol = robust_norm(vol)
    small = resize3d(vol, target)
    return small.reshape(-1), st


def paths_for(row: pd.Series, bids_dir: Path, pet_suvr_dir: Path) -> dict[str, Path]:
    sub = row["participant_id"]
    ses = row["session_id"]
    base = bids_dir / sub / ses
    return {
        "mri": base / "anat" / f"{sub}_{ses}_T1w.nii.gz",
        "pet": pet_suvr_dir / f"{sub}_{ses}_suvr.nii.gz",
        "dwi": base / "dwi" / f"{sub}_{ses}_dwi.nii.gz",
    }


def extract_split(split: str, split_path: Path, bids_dir: Path, pet_suvr_dir: Path, target: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, list[str], list[str], pd.DataFrame]:
    df = pd.read_csv(split_path, sep="\t")
    rows = []
    labels = []
    ids = []
    stat_rows = []
    for _, row in df.iterrows():
        paths = paths_for(row, bids_dir, pet_suvr_dir)
        missing = [k for k, p in paths.items() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{row['participant_id']} missing modalities: {missing}")
        feature_parts = []
        stats_all = {"participant_id": row["participant_id"], "split": split}
        for prefix, p in paths.items():
            vec, st = modality_features(p, target, prefix)
            feature_parts.append(vec)
            stats_all.update(st)
        rows.append(np.concatenate(feature_parts).astype(np.float32))
        labels.append(LABELS[row["diagnosis"]])
        ids.append(row["participant_id"])
        stat_rows.append(stats_all)
        print(f"{split}: extracted {row['participant_id']}")
    return np.vstack(rows), np.asarray(labels, dtype=np.int64), ids, df["diagnosis"].tolist(), pd.DataFrame(stat_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids-dir", default=r"D:\ALZ\data\bids")
    parser.add_argument("--pet-suvr-dir", default=r"D:\ALZ\data\pet_suvr")
    parser.add_argument("--splits-dir", default=r"D:\ALZ\data\splits")
    parser.add_argument("--out-dir", default=r"D:\ALZ\features\pilot")
    parser.add_argument("--target-size", default="16,16,16")
    args = parser.parse_args()

    target = tuple(int(x) for x in args.target_size.split(","))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bids_dir = Path(args.bids_dir)
    pet_suvr_dir = Path(args.pet_suvr_dir)
    split_dir = Path(args.splits_dir)

    all_stats = []
    metadata = {"target_size": target, "modalities": ["mri", "pet_suvr", "dwi_mean"], "labels": LABELS}
    for split in ["train", "val", "test"]:
        x, y, ids, dx, st = extract_split(split, split_dir / f"{split}.tsv", bids_dir, pet_suvr_dir, target)
        np.save(out_dir / f"{split}_features.npy", x)
        np.save(out_dir / f"{split}_labels.npy", y)
        pd.DataFrame({"participant_id": ids, "diagnosis": dx, "label": y}).to_csv(out_dir / f"{split}_ids.tsv", sep="\t", index=False)
        all_stats.append(st)
        print(f"{split}: X={x.shape}, y={y.shape}")
    pd.concat(all_stats, ignore_index=True).to_csv(out_dir / "image_stats.tsv", sep="\t", index=False)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
