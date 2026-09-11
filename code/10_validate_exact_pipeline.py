from __future__ import annotations

import argparse
from pathlib import Path


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")


def any_match(paths: list[Path], label: str) -> None:
    if not any(p.exists() for p in paths):
        pretty = ", ".join(str(p) for p in paths)
        raise SystemExit(f"Missing {label}: one of {pretty}")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate the local multimodalAD pipeline state.")
    parser.add_argument("--root", default=repo_root, type=Path)
    parser.add_argument("--require-tools", action="store_true")
    parser.add_argument("--require-trained", action="store_true")
    args = parser.parse_args()

    root = args.root
    require(root / "data" / "bids" / "participants.tsv", "participants.tsv")
    require(root / "data" / "splits" / "train.tsv", "train split")
    require(root / "data" / "splits" / "val.tsv", "val split")
    require(root / "data" / "splits" / "test.tsv", "test split")
    require(root / "data" / "MNI_cerebellum_mask.nii.gz", "cerebellum mask")

    if args.require_tools:
        require(root / "data" / "caps" / "subjects", "ClinicaDL CAPS output")
        require(root / "data" / "pet_suvr", "PET SUVR directory")
        require(root / "data" / "fod", "FOD directory")
        require(root / "code" / "multimodalAD" / "ALBEF" / "models" / "model_pretrain3D.py", "ALBEF model file")

    if args.require_trained:
        any_match(
            [
                root / "checkpoints" / "swin_fod" / "model.pt",
                root / "checkpoints" / "swin_fod" / "model_final.pt",
            ],
            "Swin-FOD checkpoint",
        )
        any_match(
            list((root / "checkpoints" / "albef").glob("checkpoint_*.pth")),
            "ALBEF checkpoint",
        )
        require(root / "features" / "fod" / "train_features.npy", "FOD train features")
        require(root / "features" / "mri_pet" / "train_features.npy", "MRI/PET train features")
        require(root / "results" / "y_pred_test.npy", "final predictions")

    print("Validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
