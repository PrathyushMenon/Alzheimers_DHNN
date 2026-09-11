"""Create a BIDS-like cohort from the strict MRI+PET+DTI subject overlap.

The script intentionally refuses to invent samples when a modality is missing.
It writes an empty participants.tsv plus a missing-modality report if the
available data cannot satisfy the multimodal cohort rule.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DIAGNOSIS_MAP = {
    "CN": "CN",
    "MCI": "MCI",
    "Dementia": "AD",
    "AD": "AD",
}


@dataclass(frozen=True)
class SeriesChoice:
    subject: str
    modality: str
    source_dir: Path
    date_hint: str


def adni_to_bids_subject(adni_id: str) -> str:
    return "sub-ADNI" + adni_id.replace("_", "")


def rid_from_adni(adni_id: str) -> int:
    return int(adni_id.split("_S_")[1])


def modality_subjects(raw_root: Path, modality: str) -> set[str]:
    base = find_adni_dir(raw_root, modality)
    if base is None:
        return set()
    return {p.name for p in base.iterdir() if p.is_dir()}


def find_adni_dir(raw_root: Path, modality: str) -> Path | None:
    direct = raw_root / modality / "ADNI"
    if direct.exists():
        return direct
    modality_root = raw_root / modality
    if not modality_root.exists():
        return None
    matches = [p for p in modality_root.rglob("ADNI") if p.is_dir()]
    if not matches:
        return None
    matches.sort(key=lambda p: len(p.parts))
    return matches[0]


def load_rda_table(path: Path) -> pd.DataFrame:
    import pyreadr

    result = pyreadr.read_r(str(path))
    if not result:
        return pd.DataFrame()
    return next(iter(result.values()))


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    if path.suffix.lower() == ".rda":
        return load_rda_table(path)
    raise ValueError(f"Unsupported clinical file format: {path}")


def find_clinical_file(raw_root: Path, names: list[str]) -> Path | None:
    search_roots = [raw_root, raw_root / "tabular", raw_root / "Tabular"]
    for root in search_roots:
        for name in names:
            candidate = root / name
            if candidate.exists():
                return candidate
            matches = sorted(root.rglob(name)) if root.exists() else []
            if matches:
                return matches[0]
            if name.lower().endswith(".csv"):
                prefix = name[:-4]
                matches = sorted(root.rglob(prefix + "*.csv")) if root.exists() else []
                if matches:
                    return matches[0]
    return None


def load_clinical(raw_root: Path) -> pd.DataFrame:
    dx_file = find_clinical_file(raw_root, ["DXSUM.rda", "DXSUM.csv"])
    demog_file = find_clinical_file(raw_root, ["PTDEMOG.csv", "PTDEMOG.rda"])
    moca_file = find_clinical_file(raw_root, ["MOCA.csv", "MOCA.rda"])
    adni_file = find_clinical_file(raw_root, ["ADNIMERGE.rda", "ADNIMERGE.csv"])

    if dx_file is None:
        raise RuntimeError("Cannot locate DXSUM clinical data under raw_root or raw_root/tabular")
    if demog_file is None:
        raise RuntimeError("Cannot locate PTDEMOG clinical data under raw_root or raw_root/tabular")
    if moca_file is None:
        raise RuntimeError("Cannot locate MOCA clinical data under raw_root or raw_root/tabular")

    dx = load_table(dx_file)
    demog = load_table(demog_file)
    moca = load_table(moca_file)

    if dx.empty:
        raise RuntimeError(f"DXSUM data is empty: {dx_file}")

    dx = dx[["RID", "VISCODE2", "EXAMDATE", "DIAGNOSIS"]].copy()
    dx["diagnosis"] = dx["DIAGNOSIS"].map(DIAGNOSIS_MAP)
    dx = dx.dropna(subset=["RID", "diagnosis"])
    dx["RID"] = dx["RID"].astype(int)
    dx = dx.sort_values(["RID", "EXAMDATE"]).drop_duplicates("RID", keep="first")

    demog_cols = ["RID", "PTGENDER", "PTDOBYY", "PTEDUCAT", "PTGENDER.1"]
    demog = demog[[c for c in demog_cols if c in demog.columns]].copy()
    demog = demog.dropna(subset=["RID"]).drop_duplicates("RID", keep="first")
    demog["RID"] = demog["RID"].astype(int)
    if "PTGENDER.1" in demog.columns and "PTGENDER" not in demog.columns:
        demog = demog.rename(columns={"PTGENDER.1": "PTGENDER"})
    demog = demog.rename(columns={"PTGENDER": "sex", "PTEDUCAT": "education"})

    moca = moca[["RID", "VISDATE", "MOCA"]].copy()
    moca["MOCA"] = pd.to_numeric(moca["MOCA"], errors="coerce")
    moca = moca.dropna(subset=["RID"]).sort_values(["RID", "MOCA", "VISDATE"], na_position="last")
    moca["RID"] = moca["RID"].astype(int)
    moca = moca.drop_duplicates("RID", keep="first").rename(columns={"MOCA": "moca"})

    clinical = dx.merge(demog, on="RID", how="left").merge(moca[["RID", "moca"]], on="RID", how="left")
    clinical["age"] = pd.NA
    if "PTDOBYY" in clinical.columns and "EXAMDATE" in clinical.columns:
        exam_year = pd.to_datetime(clinical["EXAMDATE"], errors="coerce").dt.year
        birth_year = pd.to_datetime(clinical["PTDOBYY"], errors="coerce").dt.year
        birth_year = birth_year.fillna(pd.to_numeric(clinical["PTDOBYY"], errors="coerce"))
        clinical["age"] = exam_year - birth_year
    return clinical


def choose_series(subject_dir: Path, include_keywords: tuple[str, ...], exclude_keywords: tuple[str, ...] = ()) -> SeriesChoice | None:
    candidates: list[Path] = []
    for path in subject_dir.rglob("*"):
        if not path.is_dir():
            continue
        files = [p for p in path.iterdir() if p.is_file()]
        if not files:
            continue
        lower = str(path).lower()
        if include_keywords and not any(k.lower() in lower for k in include_keywords):
            continue
        if any(k.lower() in lower for k in exclude_keywords):
            continue
        candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: str(p))
    chosen = candidates[0]
    date_hint = next((part for part in chosen.parts if len(part) >= 10 and part[4:5] == "-"), "M000")
    modality = subject_dir.parents[1].name
    return SeriesChoice(subject_dir.name, modality, chosen, date_hint)


def convert_dicom(series_dir: Path, out_dir: Path, out_stem: str) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dcm2niix = shutil.which("dcm2niix")
    if not dcm2niix:
        raise RuntimeError("dcm2niix is not available on PATH; cannot convert DICOM series.")
    cmd = [dcm2niix, "-z", "y", "-f", out_stem, "-o", str(out_dir), str(series_dir)]
    subprocess.run(cmd, check=True)
    matches = sorted(out_dir.glob(out_stem + "*.nii.gz"))
    return matches[0] if matches else None


def write_dataset_description(bids_dir: Path) -> None:
    description = {
        "Name": "Fresh ADNI multimodal cohort built from local MRI/PET/DTI overlap",
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "GeneratedBy": [{"Name": "D:/ALZ/code/01_organize_to_bids.py"}],
    }
    (bids_dir / "dataset_description.json").write_text(json.dumps(description, indent=2), encoding="utf-8")


def resolve_default_raw_root() -> Path:
    candidates = [
        Path(r"D:\ALZ\ADNI DATA"),
        Path("/mnt/d/ALZ/ADNI DATA"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default=str(resolve_default_raw_root()))
    parser.add_argument("--bids-dir", default=str(Path(__file__).resolve().parents[1] / "data" / "bids"))
    parser.add_argument("--convert", action="store_true", help="Run dcm2niix conversion. Without this, only metadata/reports are written.")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    bids_dir = Path(args.bids_dir)
    bids_dir.mkdir(parents=True, exist_ok=True)
    write_dataset_description(bids_dir)

    subjects = {m: modality_subjects(raw_root, m) for m in ["MRI", "PET", "DTI"]}
    overlap = sorted(subjects["MRI"] & subjects["PET"] & subjects["DTI"])

    missing = []
    all_subjects = sorted(set().union(*subjects.values()))
    for subject in all_subjects:
        missing.append(
            {
                "adni_id": subject,
                "has_mri": subject in subjects["MRI"],
                "has_pet": subject in subjects["PET"],
                "has_dti": subject in subjects["DTI"],
            }
        )
    pd.DataFrame(missing).to_csv(bids_dir / "missing_modalities.tsv", sep="\t", index=False)

    if not overlap:
        empty = pd.DataFrame(columns=["participant_id", "session_id", "diagnosis", "age", "sex", "moca", "adni_id", "rid"])
        empty.to_csv(bids_dir / "participants.tsv", sep="\t", index=False)
        print("No strict MRI+PET+DTI overlap found. Wrote empty participants.tsv and missing_modalities.tsv.")
        return 2

    clinical = load_clinical(raw_root)
    rows = []
    for adni_id in overlap:
        rid = rid_from_adni(adni_id)
        c = clinical[clinical["RID"] == rid]
        if c.empty:
            print(f"Skipping {adni_id}: no clinical diagnosis found.")
            continue
        participant_id = adni_to_bids_subject(adni_id)
        session_id = "ses-M000"
        row = c.iloc[0].to_dict()
        row.update({"participant_id": participant_id, "session_id": session_id, "adni_id": adni_id, "rid": rid})
        rows.append(row)

        if not args.convert:
            continue

        mri_base = find_adni_dir(raw_root, "MRI")
        pet_base = find_adni_dir(raw_root, "PET")
        dti_base = find_adni_dir(raw_root, "DTI")
        if not (mri_base and pet_base and dti_base):
            raise RuntimeError("Could not locate nested ADNI folders for MRI, PET, and DTI.")
        mri = choose_series(mri_base / adni_id, ("mp-rage", "mprage"), ("repeat",))
        pet = choose_series(pet_base / adni_id, ("brain", "av1451", "tau", "ftp", "flortaucipir"), ())
        dwi = choose_series(dti_base / adni_id, ("dti", "diffusion"), ("adc", "fa", "tracew"))
        if not (mri and pet and dwi):
            print(f"Skipping conversion for {adni_id}: one or more series choices missing.")
            continue

        anat_dir = bids_dir / participant_id / session_id / "anat"
        pet_dir = bids_dir / participant_id / session_id / "pet"
        dwi_dir = bids_dir / participant_id / session_id / "dwi"
        convert_dicom(mri.source_dir, anat_dir, f"{participant_id}_{session_id}_T1w")
        convert_dicom(pet.source_dir, pet_dir, f"{participant_id}_{session_id}_trc-av1451_pet")
        convert_dicom(dwi.source_dir, dwi_dir, f"{participant_id}_{session_id}_dwi")

    participants = pd.DataFrame(rows)
    keep = ["participant_id", "session_id", "diagnosis", "age", "sex", "moca", "adni_id", "rid"]
    participants = participants[[c for c in keep if c in participants.columns]]
    participants.to_csv(bids_dir / "participants.tsv", sep="\t", index=False)
    print(f"Final strict cohort size: {len(participants)}")
    if not participants.empty:
        print(participants["diagnosis"].value_counts(dropna=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
