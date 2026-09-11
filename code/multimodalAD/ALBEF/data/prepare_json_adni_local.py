from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


DIAGNOSIS_TO_CDR = {
    "CN": 0.0,
    "MCI": 0.5,
    "EMCI": 0.5,
    "LMCI": 0.5,
    "AD": 2.0,
}


def find_t1(caps_dir: Path, participant: str, session: str) -> Path:
    root = caps_dir / "subjects" / participant / session / "t1_linear"
    matches = sorted(root.glob("*_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz"))
    if not matches:
        matches = sorted(root.glob("*_T1w.nii.gz"))
    if not matches:
        raise FileNotFoundError(f"Missing ClinicaDL t1-linear MRI for {participant}/{session}: {root}")
    return matches[0]


def find_suvr(pet_suvr_dir: Path, participant: str, session: str) -> Path:
    path = pet_suvr_dir / f"{participant}_{session}_suvr.nii.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing PET SUVR for {participant}/{session}: {path}")
    return path


def ensure_neutral_ref_dat(pet_path: Path) -> None:
    # The official ALBEF loader divides tau_pet by a sibling
    # km_inferior.ref.tac.dat file. Our tau_pet input is already SUVR, so the
    # neutral reference keeps official loader behavior without double scaling.
    dat_path = pet_path.parent / "km_inferior.ref.tac.dat"
    dat_path.write_text("1.00000000\n", encoding="utf-8")


def build_rows(split_path: Path, caps_dir: Path, pet_suvr_dir: Path) -> list[dict]:
    df = pd.read_csv(split_path, sep="\t", dtype=str)
    required = {"participant_id", "session_id", "diagnosis"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{split_path} missing required columns: {sorted(missing)}")

    rows: list[dict] = []
    for _, row in df.iterrows():
        diagnosis = row["diagnosis"].strip().upper()
        if diagnosis not in DIAGNOSIS_TO_CDR:
            raise ValueError(f"Unsupported diagnosis {diagnosis!r} in {split_path}")
        mri = find_t1(caps_dir, row["participant_id"], row["session_id"])
        pet = find_suvr(pet_suvr_dir, row["participant_id"], row["session_id"])
        ensure_neutral_ref_dat(pet)
        rows.append({"mri": mri.as_posix(), "tau_pet": pet.as_posix(), "cdr": DIAGNOSIS_TO_CDR[diagnosis], "amyloid_pet": ""})
    return rows


def main() -> int:
    repo_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description="Prepare exact local ALBEF JSON annotations from CAPS, SUVR, and fixed splits.")
    parser.add_argument("--caps-dir", default=repo_root / "data" / "caps", type=Path)
    parser.add_argument("--pet-suvr-dir", default=repo_root / "data" / "pet_suvr", type=Path)
    parser.add_argument("--splits-dir", default=repo_root / "data" / "splits", type=Path)
    parser.add_argument("--out-dir", default=Path(__file__).resolve().parent, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        rows = build_rows(args.splits_dir / f"{split}.tsv", args.caps_dir, args.pet_suvr_dir)
        out = args.out_dir / f"ADNI_{split}_local.json"
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        # test_pretrain3D.yaml in the repo historically points at non-local
        # names. Keep both names synced to avoid accidental stale configs.
        shutil.copyfile(out, args.out_dir / f"ADNI_{split}.json")
        print(f"Wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
