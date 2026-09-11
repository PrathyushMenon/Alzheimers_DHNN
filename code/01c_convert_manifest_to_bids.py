"""Convert exact manifest image IDs from ADNI DICOM export into BIDS-like layout."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def adni_to_bids_subject(adni_id: str) -> str:
    return "sub-ADNI" + adni_id.replace("_", "")


def find_adni_dir(raw_root: Path, modality: str) -> Path:
    direct = raw_root / modality / "ADNI"
    if direct.exists():
        return direct
    matches = [p for p in (raw_root / modality).rglob("ADNI") if p.is_dir()]
    if not matches:
        raise FileNotFoundError(f"Could not locate ADNI folder under {raw_root / modality}")
    matches.sort(key=lambda p: len(p.parts))
    return matches[0]


def find_image_dir(adni_base: Path, subject: str, image_id: int | str) -> Path:
    target = f"I{int(image_id)}"
    subject_dir = adni_base / subject
    matches = [p for p in subject_dir.rglob(target) if p.is_dir()]
    if not matches:
        raise FileNotFoundError(f"Could not find {target} under {subject_dir}")
    matches.sort(key=lambda p: len(p.parts))
    return matches[0]


def run_dcm2niix(src: Path, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if list(out_dir.glob(stem + "*.nii*")):
        print(f"SKIP existing {stem}")
        return
    dcm2niix = shutil.which("dcm2niix")
    if not dcm2niix:
        raise RuntimeError("dcm2niix is not available on PATH.")
    cmd = [dcm2niix, "-z", "y", "-b", "y", "-f", stem, "-o", str(out_dir), str(src)]
    print("RUN", " ".join(cmd))
    subprocess.run(cmd, check=True)


def write_dataset_description(bids_dir: Path) -> None:
    bids_dir.mkdir(parents=True, exist_ok=True)
    description = {
        "Name": "ADNI MRI PET DTI exact-image-ID cohort",
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "GeneratedBy": [{"Name": "01c_convert_manifest_to_bids.py"}],
    }
    (bids_dir / "dataset_description.json").write_text(json.dumps(description, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default=r"D:\ALZ\ADNI DATA")
    parser.add_argument("--manifest", default=r"D:\ALZ\results\combined_33_plus_ad_manifest.csv")
    parser.add_argument("--participants", default=r"D:\ALZ\data\bids\participants.tsv")
    parser.add_argument("--bids-dir", default=r"D:\ALZ\data\bids")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    bids_dir = Path(args.bids_dir)
    write_dataset_description(bids_dir)

    manifest = pd.read_csv(args.manifest)
    participants = pd.read_csv(args.participants, sep="\t")
    keep = set(participants["adni_id"])
    manifest = manifest[manifest["Subject ID"].isin(keep)].copy()
    if args.limit:
        manifest = manifest.head(args.limit)

    bases = {m: find_adni_dir(raw_root, m) for m in ["MRI", "PET", "DTI"]}
    errors: list[dict[str, str]] = []
    for _, row in manifest.iterrows():
        adni_id = row["Subject ID"]
        sub = adni_to_bids_subject(adni_id)
        ses = "ses-M000"
        try:
            mri_src = find_image_dir(bases["MRI"], adni_id, row["mri_image_id"])
            pet_src = find_image_dir(bases["PET"], adni_id, row["pet_image_id"])
            dti_src = find_image_dir(bases["DTI"], adni_id, row["dti_image_id"])
            run_dcm2niix(mri_src, bids_dir / sub / ses / "anat", f"{sub}_{ses}_T1w")
            run_dcm2niix(pet_src, bids_dir / sub / ses / "pet", f"{sub}_{ses}_trc-av1451_pet")
            run_dcm2niix(dti_src, bids_dir / sub / ses / "dwi", f"{sub}_{ses}_dwi")
        except Exception as exc:
            print(f"ERROR {adni_id}: {exc}")
            errors.append({"adni_id": adni_id, "error": str(exc)})

    if errors:
        pd.DataFrame(errors).to_csv(bids_dir / "conversion_errors.tsv", sep="\t", index=False)
        print(f"Conversion completed with {len(errors)} errors.")
        return 1
    print(f"Conversion completed for {len(manifest)} subjects.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
