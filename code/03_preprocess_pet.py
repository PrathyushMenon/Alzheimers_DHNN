"""Paper-faithful tau PET preprocessing.

This stage intentionally has no geometric or synthetic fallback. The paper
requires PET in the same T1/MNI space and SUVR using a cerebellar reference
region. If ANTs, ClinicaDL T1-linear output, or a real cerebellar mask is
missing, the script fails so the reproduction cannot silently drift.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

try:
    import ants  # type: ignore
except ImportError as exc:  # pragma: no cover - environment dependent
    ants = None
    ANTS_IMPORT_ERROR = exc
else:
    ANTS_IMPORT_ERROR = None


def find_t1_reference(caps_dir: Path, sub: str, ses: str) -> Path:
    root = caps_dir / "subjects" / sub / ses / "t1_linear"
    matches = sorted(root.glob("*_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz"))
    if not matches:
        matches = sorted(root.glob("*_T1w.nii.gz"))
    if not matches:
        raise FileNotFoundError(
            f"No ClinicaDL t1-linear T1w reference for {sub} {ses}. "
            f"Expected under {root}."
        )
    return matches[0]


def register_pet_to_t1(pet_path: Path, t1_path: Path) -> tuple[nib.Nifti1Image, Path]:
    if ants is None:
        raise RuntimeError(f"antspyx/ANTs is required for PET registration: {ANTS_IMPORT_ERROR}")
    fixed = ants.image_read(str(t1_path))
    moving = ants.image_read(str(pet_path))
    reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="Rigid")
    tmp = Path(tempfile.gettempdir()) / f"{pet_path.stem}_rigid_to_t1.nii.gz"
    ants.image_write(reg["warpedmovout"], str(tmp))
    return nib.load(str(tmp)), tmp


def resample_mask_to_pet(mask_path: Path, pet_path: Path) -> np.ndarray:
    if ants is None:
        raise RuntimeError(f"antspyx/ANTs is required for mask resampling: {ANTS_IMPORT_ERROR}")
    fixed = ants.image_read(str(pet_path))
    moving = ants.image_read(str(mask_path))
    mask = ants.resample_image_to_target(moving, fixed, interp_type="nearestNeighbor")
    return mask.numpy().astype(bool)


def compute_suvr(pet_img: nib.Nifti1Image, mask: np.ndarray) -> nib.Nifti1Image:
    data = pet_img.get_fdata(dtype=np.float32)
    if mask.shape != data.shape[:3]:
        raise ValueError(f"Mask shape {mask.shape} does not match PET shape {data.shape[:3]}")
    ref = data[mask]
    ref = ref[np.isfinite(ref) & (ref > 0)]
    if ref.size == 0:
        raise ValueError("Cerebellar mask contains no positive finite PET voxels.")
    suvr = (data / float(ref.mean())).astype(np.float32)
    return nib.Nifti1Image(suvr, pet_img.affine, pet_img.header)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Register AV1451 PET to ClinicaDL T1w space and compute SUVR.")
    parser.add_argument("--bids-dir", default=repo_root / "data" / "bids", type=Path)
    parser.add_argument("--caps-dir", default=repo_root / "data" / "caps", type=Path)
    parser.add_argument("--out-dir", default=repo_root / "data" / "pet_suvr", type=Path)
    parser.add_argument("--mask", default=repo_root / "data" / "MNI_cerebellum_mask.nii.gz", type=Path)
    args = parser.parse_args()

    if ants is None:
        raise SystemExit(f"ERROR: antspyx/ANTs is required for exact PET preprocessing: {ANTS_IMPORT_ERROR}")
    if not args.mask.exists():
        raise SystemExit(
            "ERROR: real MNI cerebellar reference mask is required. "
            f"Expected: {args.mask}. Do not use a synthetic fallback mask."
        )

    pet_files = sorted(args.bids_dir.glob("sub-*/ses-*/pet/*pet.nii.gz"))
    if not pet_files:
        raise SystemExit(f"ERROR: no BIDS PET NIfTI files found under {args.bids_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for pet_path in pet_files:
        sub = next(part for part in pet_path.parts if part.startswith("sub-"))
        ses = next(part for part in pet_path.parts if part.startswith("ses-"))
        t1_path = find_t1_reference(args.caps_dir, sub, ses)
        pet_reg, pet_reg_path = register_pet_to_t1(pet_path, t1_path)
        mask = resample_mask_to_pet(args.mask, pet_reg_path)
        suvr_img = compute_suvr(pet_reg, mask)
        out_path = args.out_dir / f"{sub}_{ses}_suvr.nii.gz"
        nib.save(suvr_img, str(out_path))

        # The official ALBEF dataloader divides by this sidecar. Since this
        # file is already SUVR, the exact local annotation uses a neutral ref.
        np.savetxt(args.out_dir / f"{sub}_{ses}_km_inferior.ref.tac.dat", [1.0], fmt="%.8f")
        written += 1
        print(f"Wrote {out_path}")

    print(f"PET SUVR complete: {written}/{len(pet_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
