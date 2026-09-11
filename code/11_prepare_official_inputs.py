from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


LABEL_TO_OFFICIAL = {"CN": 1, "MCI": 2, "EMCI": 2, "LMCI": 2, "AD": 3}
LABEL_TO_CDR = {"CN": 0.0, "MCI": 0.5, "EMCI": 0.5, "LMCI": 0.5, "AD": 2.0}


def find_t1(caps_dir: Path, sub: str, ses: str) -> Path:
    root = caps_dir / "subjects" / sub / ses / "t1_linear"
    matches = sorted(root.glob("*_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz"))
    if not matches:
        matches = sorted(root.glob("*_T1w.nii.gz"))
    if not matches:
        raise FileNotFoundError(f"Missing t1-linear MRI for {sub}/{ses}: {root}")
    return matches[0]


def find_suvr(pet_suvr_dir: Path, sub: str, ses: str) -> Path:
    path = pet_suvr_dir / f"{sub}_{ses}_suvr.nii.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing PET SUVR for {sub}/{ses}: {path}")
    (path.parent / "km_inferior.ref.tac.dat").write_text("1.00000000\n", encoding="utf-8")
    return path


def find_fod(fod_dir: Path, sub: str, ses: str) -> Path:
    path = fod_dir / f"{sub}_{ses}" / "fod_lmax4_MNI.nii.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing MNI FOD for {sub}/{ses}: {path}")
    return path


def load_split(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    df["split"] = split
    return df


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate official-repo local manifests for Swin-FOD, ALBEF, and TabPFN.")
    parser.add_argument("--caps-dir", default=root / "data" / "caps", type=Path)
    parser.add_argument("--pet-suvr-dir", default=root / "data" / "pet_suvr", type=Path)
    parser.add_argument("--fod-dir", default=root / "data" / "fod", type=Path)
    parser.add_argument("--splits-dir", default=root / "data" / "splits", type=Path)
    parser.add_argument("--out-dir", default=root / "data" / "official_inputs", type=Path)
    parser.add_argument("--albef-data-dir", default=root / "code" / "multimodalAD" / "ALBEF" / "data", type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.albef_data_dir.mkdir(parents=True, exist_ok=True)

    frames = [load_split(args.splits_dir / f"{split}.tsv", split) for split in ("train", "val", "test")]
    df = pd.concat(frames, ignore_index=True)

    swin_rows = []
    albef_by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for _, row in df.iterrows():
        sub, ses = row["participant_id"], row["session_id"]
        diagnosis = row["diagnosis"].strip().upper()
        if diagnosis not in LABEL_TO_OFFICIAL:
            raise ValueError(f"Unsupported diagnosis {diagnosis!r} for {sub}/{ses}")
        t1 = find_t1(args.caps_dir, sub, ses).as_posix()
        pet = find_suvr(args.pet_suvr_dir, sub, ses).as_posix()
        fod = find_fod(args.fod_dir, sub, ses).as_posix()
        label = LABEL_TO_OFFICIAL[diagnosis]
        swin_rows.append(
            {
                "Subject": sub.replace("sub-ADNI", ""),
                "Exam Date": "1/1/2000",
                "Diagnosis": label,
                "split": row["split"],
                "label": label,
                "t1": t1,
                "taupet": pet,
                "fod": fod,
            }
        )
        albef_by_split[row["split"]].append({"mri": t1, "tau_pet": pet, "cdr": LABEL_TO_CDR[diagnosis], "amyloid_pet": ""})

    swin_csv = args.out_dir / "swin_fod_local.csv"
    pd.DataFrame(swin_rows).to_csv(swin_csv, index=False)
    print(f"Wrote Swin-FOD manifest: {swin_csv} ({len(swin_rows)} rows)")

    for split, rows in albef_by_split.items():
        for name in (f"ADNI_{split}_local.json", f"ADNI_{split}.json"):
            out = args.albef_data_dir / name
            out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(f"Wrote ALBEF manifest: {out} ({len(rows)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
