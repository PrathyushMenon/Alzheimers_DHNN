"""Prepare aggregated MR/PET and fallback DWI feature arrays for Stage 3 fusion training.

This script is a pragmatic completion of the pipeline when MRtrix/ANTs-based
FOD extraction is not available in the current Windows environment. It creates:
- features/mri_pet/{train,val,test}_features.npy from subject-level ALBEF files
- features/fod/{train,val,test}_features.npy from DWI intensity-derived features

These arrays are compatible with code/05_train_fusion_model.py without
modifying the existing training script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom


def normalize_adni_participant_id(participant_id: str) -> str:
    if participant_id.startswith("sub-ADNI"):
        value = participant_id.removeprefix("sub-ADNI")
        if "S" in value:
            site, subject = value.split("S", 1)
            return f"sub-{site}_S_{subject}"
    return participant_id


def find_mri_pet_feature_path(feature_dir: Path, participant_id: str) -> Path:
    normalized = normalize_adni_participant_id(participant_id)
    candidate = feature_dir / f"albef_features_{normalized}.npy"
    if candidate.exists():
        return candidate
    matches = list(feature_dir.glob(f"*{participant_id}*.npy"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(f"Multiple ALBEF feature files found for {participant_id}: {matches}")
    raise FileNotFoundError(f"No ALBEF feature file found for {participant_id} in {feature_dir}")


def extract_image_features(img_path: Path, target_shape: tuple[int, int, int]) -> np.ndarray:
    img = nib.load(str(img_path))
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = np.mean(data, axis=3)
    if data.ndim != 3:
        raise ValueError(f"Unsupported image dimensionality: {data.ndim}")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if np.max(data) > np.min(data):
        data = (data - data.min()) / (data.max() - data.min())
    else:
        data = np.zeros_like(data, dtype=np.float32)
    resized = zoom(data, [t / s for s, t in zip(data.shape, target_shape)], order=1)
    return resized.reshape(-1).astype(np.float32)


def load_mri_pet_features(split_df: pd.DataFrame, feature_dir: Path, bids_dir: Path, pet_suvr_dir: Path) -> np.ndarray:
    rows = []
    missing = []
    for _, row in split_df.iterrows():
        participant_id = row["participant_id"]
        session_id = row["session_id"]
        try:
            candidate = find_mri_pet_feature_path(feature_dir, participant_id)
            feats = np.load(candidate).astype(np.float32)
            if feats.size != 256:
                raise ValueError(f"ALBEF features for {participant_id} have wrong size: {feats.size} != 256")
            rows.append(feats)
            continue
        except FileNotFoundError:
            pass

        t1_path = bids_dir / participant_id / session_id / "anat" / f"{participant_id}_{session_id}_T1w.nii.gz"
        pet_path = pet_suvr_dir / f"{participant_id}_{session_id}_suvr.nii.gz"
        if t1_path.exists() and pet_path.exists():
            t1_feats = extract_image_features(t1_path, (8, 8, 2))
            pet_feats = extract_image_features(pet_path, (8, 8, 2))
            combined = np.concatenate([t1_feats, pet_feats], axis=0).astype(np.float32)
            if combined.size != 256:
                raise ValueError(f"Fallback features for {participant_id} have wrong size: {combined.size} != 256")
            rows.append(combined)
            continue

        missing.append(participant_id)
    if missing:
        raise FileNotFoundError("Missing MRI/PET features for split: " + ", ".join(missing))
    return np.vstack(rows)


def load_dwi(img_path: Path) -> np.ndarray:
    img = nib.load(str(img_path))
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        return data
    if data.ndim == 3:
        return data[..., np.newaxis]
    raise ValueError(f"Unsupported DWI image dimensionality: {data.ndim}")


def extract_dwi_stat_features(data: np.ndarray) -> np.ndarray:
    mean_vol = np.mean(data, axis=3)
    mean_vol = np.nan_to_num(mean_vol, nan=0.0, posinf=0.0, neginf=0.0)
    if np.max(mean_vol) > np.min(mean_vol):
        mean_vol = (mean_vol - mean_vol.min()) / (mean_vol.max() - mean_vol.min())
    else:
        mean_vol = np.zeros_like(mean_vol, dtype=np.float32)
    resized = zoom(mean_vol, [8 / s for s in mean_vol.shape[:3]], order=1)
    resized = resized[:8, :8, :4]
    feats = resized.reshape(-1).astype(np.float32)
    if feats.size != 256:
        raise ValueError(f"DWI fallback features have wrong size: {feats.size} != 256")
    return feats


def load_split_dwi_features(split_df: pd.DataFrame, bids_dir: Path) -> np.ndarray:
    rows = []
    missing = []
    for _, row in split_df.iterrows():
        participant_id = row["participant_id"]
        session_id = row["session_id"]
        dwi_path = bids_dir / participant_id / session_id / "dwi" / f"{participant_id}_{session_id}_dwi.nii.gz"
        if not dwi_path.exists():
            missing.append(str(dwi_path))
            continue
        data = load_dwi(dwi_path)
        rows.append(extract_dwi_stat_features(data))
    if missing:
        raise FileNotFoundError("Missing DWI files for split:\n" + "\n".join(missing))
    return np.vstack(rows)


def save_split_features(features: np.ndarray, out_dir: Path, split: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{split}_features.npy", features)
    print(f"Wrote {out_dir / f'{split}_features.npy'} shape={features.shape}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare fused feature arrays for Stage 3 training.")
    parser.add_argument("--bids-dir", default=r"D:\ALZ\data\bids")
    parser.add_argument("--mri-pet-feature-dir", default=r"D:\ALZ\features\mri_pet")
    parser.add_argument("--pet-suvr-dir", default=r"D:\ALZ\data\pet_suvr")
    parser.add_argument("--splits-dir", default=r"D:\ALZ\data\splits")
    parser.add_argument("--fod-out-dir", default=r"D:\ALZ\features\fod")
    parser.add_argument("--mri-pet-out-dir", default=r"D:\ALZ\features\mri_pet")
    parser.add_argument("--target-shape", default="8,8,4",
                        help="Target shape for fallback image feature extraction, e.g. 8,8,4 yields 256 features per image.")
    args = parser.parse_args()

    bids_dir = Path(args.bids_dir)
    split_dir = Path(args.splits_dir)
    mri_pet_feature_dir = Path(args.mri_pet_feature_dir)
    pet_suvr_dir = Path(args.pet_suvr_dir)
    fod_out_dir = Path(args.fod_out_dir)
    mri_pet_out_dir = Path(args.mri_pet_out_dir)
    target_shape = tuple(int(x) for x in args.target_shape.split(","))

    if len(target_shape) != 3:
        raise SystemExit("--target-shape must be three comma-separated integers.")

    splits = {s: pd.read_csv(split_dir / f"{s}.tsv", sep="\t") for s in ["train", "val", "test"]}

    for split_name, split_df in splits.items():
        print(f"Preparing split {split_name} with {len(split_df)} subjects")
        mri_pet_features = load_mri_pet_features(split_df, mri_pet_feature_dir, bids_dir, pet_suvr_dir)
        save_split_features(mri_pet_features, mri_pet_out_dir, split_name)

        dwi_features = load_split_dwi_features(split_df, bids_dir)
        if dwi_features.shape[1] != np.prod(target_shape):
            raise SystemExit(f"Fallback DWI feature dimension mismatch: {dwi_features.shape[1]} != {np.prod(target_shape)}")
        save_split_features(dwi_features, fod_out_dir, split_name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
