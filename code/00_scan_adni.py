"""Audit the local ADNI download before any preprocessing.

This script follows Step 0 from AGENT_PROMPT.md, but uses Windows paths by
default because this workspace is running from PowerShell. It only reads from
the ADNI data folder and writes reports under this project.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
from pathlib import Path
from typing import Iterable


SUBJECT_REPR_LEN = 20


def subject_dirs(root: Path, modality: str) -> set[str]:
    base = find_adni_dir(root, modality)
    if base is None:
        return set()
    return {p.name for p in base.iterdir() if p.is_dir()}


def find_adni_dir(root: Path, modality: str) -> Path | None:
    """Return the ADNI subject directory for a modality across common exports."""
    direct = root / modality / "ADNI"
    if direct.exists():
        return direct
    modality_root = root / modality
    if not modality_root.exists():
        return None
    matches = [p for p in modality_root.rglob("ADNI") if p.is_dir()]
    if not matches:
        return None
    matches.sort(key=lambda p: len(p.parts))
    return matches[0]


def first_files(root: Path, patterns: Iterable[str], limit: int, max_dirs: int = 2000) -> list[str]:
    hits: list[str] = []
    visited = 0
    for dirpath, _, filenames in os.walk(root):
        visited += 1
        if visited > max_dirs:
            break
        for filename in filenames:
            if any(fnmatch.fnmatch(filename, pattern) for pattern in patterns):
                hits.append(str(Path(dirpath) / filename))
                if len(hits) >= limit:
                    return hits
    return hits


def count_files(root: Path, patterns: Iterable[str]) -> int:
    total = 0
    for pattern in patterns:
        total += sum(1 for p in root.rglob(pattern) if p.is_file())
    return total


def clinical_tables(root: Path) -> dict[str, str]:
    names = ["MOCA.rda", "PTDEMOG.rda", "DXSUM.rda", "ADSL.rda", "MMSE_16May2026.csv"]
    found: dict[str, str] = {}
    for name in names:
        matches = list(root.rglob(name))
        if matches:
            found[name] = str(matches[0])
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default=r"D:\alzhimers\ADNI DATA")
    parser.add_argument("--out-dir", default=r"D:\ALZ\results")
    parser.add_argument("--deep-count", action="store_true", help="Count all matching files; slow on large DICOM trees.")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_root.exists():
        raise SystemExit(f"Raw ADNI root does not exist: {raw_root}")

    subjects = {m: subject_dirs(raw_root, m) for m in ["MRI", "PET", "DTI"]}
    mri_pet = subjects["MRI"] & subjects["PET"]
    mri_dti = subjects["MRI"] & subjects["DTI"]
    pet_dti = subjects["PET"] & subjects["DTI"]
    overlap = subjects["MRI"] & subjects["PET"] & subjects["DTI"]

    report = {
        "raw_root": str(raw_root),
        "counts": {
            "N_MRI": len(subjects["MRI"]),
            "N_PET": len(subjects["PET"]),
            "N_DTI": len(subjects["DTI"]),
            "N_overlap": len(overlap),
            "N_MRI_PET": len(mri_pet),
            "N_MRI_DTI": len(mri_dti),
            "N_PET_DTI": len(pet_dti),
        },
        "overlap_subjects": sorted(overlap),
        "pairwise_overlap": {
            "MRI_PET": sorted(mri_pet),
            "MRI_DTI": sorted(mri_dti),
            "PET_DTI": sorted(pet_dti),
        },
        "examples": {
            "nifti": first_files(raw_root, ["*.nii", "*.nii.gz"], 20),
            "dicom": first_files(raw_root, ["*.dcm", "*.IMA", "I*"], 20),
            "csv": first_files(raw_root, ["*.csv"], 20),
        },
        "file_counts": (
            {
                "nifti": count_files(raw_root, ["*.nii", "*.nii.gz"]),
                "dicom_like": count_files(raw_root, ["*.dcm", "*.IMA", "I*"]),
                "csv": count_files(raw_root, ["*.csv"]),
            }
            if args.deep_count
            else {"status": "not_computed", "reason": "Use --deep-count to scan the full DICOM tree."}
        ),
        "clinical_tables": clinical_tables(raw_root),
    }

    json_path = out_dir / "data_audit.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_path = out_dir / "data_audit.md"
    lines = [
        "# ADNI Data Audit",
        "",
        f"Raw root: `{raw_root}`",
        "",
        "## Modality Counts",
        "",
        f"- N_MRI={report['counts']['N_MRI']}",
        f"- N_PET={report['counts']['N_PET']}",
        f"- N_DTI={report['counts']['N_DTI']}",
        f"- N_overlap={report['counts']['N_overlap']}",
        f"- N_MRI_PET={report['counts']['N_MRI_PET']}",
        f"- N_MRI_DTI={report['counts']['N_MRI_DTI']}",
        f"- N_PET_DTI={report['counts']['N_PET_DTI']}",
        "",
        "## Three-Modality Overlap",
        "",
    ]
    if overlap:
        lines.extend(f"- {s}" for s in sorted(overlap)[:SUBJECT_REPR_LEN])
    else:
        lines.append("No subject currently has MRI + PET + DTI in the local folders.")
    lines.extend(
        [
            "",
            "## Clinical Tables Found",
            "",
        ]
    )
    lines.extend(f"- {k}: `{v}`" for k, v in report["clinical_tables"].items())
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        "N_MRI={N_MRI}, N_PET={N_PET}, N_DTI={N_DTI}, N_overlap={N_overlap}".format(
            **report["counts"]
        )
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    if not overlap:
        print("BLOCKER: no MRI+PET+DTI subject overlap is available for multimodal training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
