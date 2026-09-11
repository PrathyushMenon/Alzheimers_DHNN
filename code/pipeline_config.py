from __future__ import annotations

import os
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


WORKSPACE_ROOT = Path(os.environ.get("ALZ_ROOT", _repo_root())).resolve()
RAW_DATA_ROOT = Path(os.environ.get("RAW_ROOT", WORKSPACE_ROOT / "ADNI DATA")).resolve()
BIDS_ROOT = Path(os.environ.get("BIDS_DIR", WORKSPACE_ROOT / "data" / "bids")).resolve()
CAPS_ROOT = Path(os.environ.get("CAPS_DIR", WORKSPACE_ROOT / "data" / "caps")).resolve()
PET_SUVR_ROOT = Path(os.environ.get("PET_SUVR_DIR", WORKSPACE_ROOT / "data" / "pet_suvr")).resolve()
FOD_ROOT = Path(os.environ.get("FOD_DIR", WORKSPACE_ROOT / "data" / "fod")).resolve()
SPLITS_ROOT = Path(os.environ.get("SPLITS_DIR", WORKSPACE_ROOT / "data" / "splits")).resolve()
TABULAR_ROOT = Path(os.environ.get("TABULAR_DIR", WORKSPACE_ROOT / "ADNI DATA" / "tabular")).resolve()
CHECKPOINTS_ROOT = Path(os.environ.get("CHECKPOINTS_DIR", WORKSPACE_ROOT / "checkpoints")).resolve()
FEATURES_ROOT = Path(os.environ.get("FEATURES_DIR", WORKSPACE_ROOT / "features")).resolve()
RESULTS_ROOT = Path(os.environ.get("RESULTS_DIR", WORKSPACE_ROOT / "results")).resolve()
LOGS_ROOT = Path(os.environ.get("LOGS_DIR", WORKSPACE_ROOT / "logs")).resolve()
MULTIMODAL_REPO = Path(os.environ.get("MULTIMODAL_REPO", WORKSPACE_ROOT / "code" / "multimodalAD")).resolve()

ALBEF_PRETRAIN_CHECKPOINT = Path(
    os.environ.get("ALBEF_PRETRAIN_CHECKPOINT", CHECKPOINTS_ROOT / "albef" / "hable_pretrain_checkpoint.pth")
).resolve()
CEREBELLUM_MASK = Path(os.environ.get("CEREBELLUM_MASK", WORKSPACE_ROOT / "data" / "MNI_cerebellum_mask.nii.gz")).resolve()
