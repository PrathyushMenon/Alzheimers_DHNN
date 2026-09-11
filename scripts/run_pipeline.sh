#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python3}"
RAW_ROOT="${RAW_ROOT:-$ROOT_DIR/ADNI DATA}"
BIDS_DIR="${BIDS_DIR:-$ROOT_DIR/data/bids}"
CAPS_DIR="${CAPS_DIR:-$ROOT_DIR/data/caps}"
SPLITS_DIR="${SPLITS_DIR:-$ROOT_DIR/data/splits}"
PET_SUVR_DIR="${PET_SUVR_DIR:-$ROOT_DIR/data/pet_suvr}"
FOD_DIR="${FOD_DIR:-$ROOT_DIR/data/fod}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
CEREBELLUM_MASK="${CEREBELLUM_MASK:-$ROOT_DIR/data/MNI_cerebellum_mask.nii.gz}"
ALBEF_PRETRAIN_CHECKPOINT="${ALBEF_PRETRAIN_CHECKPOINT:-$ROOT_DIR/checkpoints/albef/hable_pretrain_checkpoint.pth}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$LOG_DIR" "$ROOT_DIR/checkpoints/swin_fod" "$ROOT_DIR/checkpoints/albef" "$ROOT_DIR/features/fod" "$ROOT_DIR/features/mri_pet" "$ROOT_DIR/results"

run_stage() {
  local title="$1"
  shift
  echo
  echo "=== $title ==="
  "$@" 2>&1 | tee "$LOG_DIR/$title.log"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required command not found on PATH: $1" >&2
    exit 2
  fi
}

echo "=== Exact multimodalAD reproduction pipeline ==="
echo "ROOT_DIR=$ROOT_DIR"
echo "RAW_ROOT=$RAW_ROOT"
echo "CEREBELLUM_MASK=$CEREBELLUM_MASK"
echo "ALBEF_PRETRAIN_CHECKPOINT=$ALBEF_PRETRAIN_CHECKPOINT"

require_cmd "$PYTHON"

run_stage "00_validate_inputs_initial" "$PYTHON" code/10_validate_exact_pipeline.py --root "$ROOT_DIR" || true

if [ ! -f "$BIDS_DIR/participants.tsv" ]; then
  run_stage "01_organize_to_bids" "$PYTHON" code/01_organize_to_bids.py --raw-root "$RAW_ROOT" --bids-dir "$BIDS_DIR" --convert
fi

if [ ! -f "$SPLITS_DIR/train.tsv" ] || [ ! -f "$SPLITS_DIR/val.tsv" ] || [ ! -f "$SPLITS_DIR/test.tsv" ]; then
  run_stage "02_make_splits" "$PYTHON" code/02_make_splits.py --participants "$BIDS_DIR/participants.tsv" --out-dir "$SPLITS_DIR"
fi

if [ ! -d "$CAPS_DIR/subjects" ]; then
  require_cmd clinicadl
  run_stage "03_clinicadl_t1_linear" clinicadl preprocessing run t1-linear "$BIDS_DIR" "$CAPS_DIR"
fi

require_cmd antsRegistration
if [ ! -f "$CEREBELLUM_MASK" ]; then
  echo "ERROR: missing real cerebellar reference mask: $CEREBELLUM_MASK" >&2
  echo "Download/create the ADNI tau PET inferior cerebellar GM mask in MNI space, then rerun." >&2
  exit 3
fi
run_stage "04_pet_suvr" "$PYTHON" code/03_preprocess_pet.py --bids-dir "$BIDS_DIR" --caps-dir "$CAPS_DIR" --out-dir "$PET_SUVR_DIR" --mask "$CEREBELLUM_MASK"

require_cmd dwidenoise
require_cmd dwifslpreproc
require_cmd dwi2fod
require_cmd mrtransform
run_stage "05_compute_fod" bash code/04_compute_fod.sh

run_stage "06_prepare_official_inputs" "$PYTHON" code/11_prepare_official_inputs.py --caps-dir "$CAPS_DIR" --pet-suvr-dir "$PET_SUVR_DIR" --fod-dir "$FOD_DIR" --splits-dir "$SPLITS_DIR"

run_stage "07_validate_pretraining_ready" "$PYTHON" code/10_validate_exact_pipeline.py --root "$ROOT_DIR" --require-tools
if [ ! -f "$ALBEF_PRETRAIN_CHECKPOINT" ]; then
  echo "ERROR: ALBEF training requires the paper/HABLE pretrained checkpoint." >&2
  echo "Expected: $ALBEF_PRETRAIN_CHECKPOINT" >&2
  echo "Put the checkpoint there or set ALBEF_PRETRAIN_CHECKPOINT=/path/to/checkpoint.pth." >&2
  exit 4
fi
run_stage "08_train_swin_fod" bash scripts/train_swin_fod.sh "$ROOT_DIR/checkpoints/swin_fod"

SWIN_CKPT="$(find "$ROOT_DIR/checkpoints/swin_fod" \( -name model.pt -o -name model_final.pt \) | head -1)"
if [ -z "${SWIN_CKPT:-}" ]; then
  echo "ERROR: Swin-FOD checkpoint was not produced." >&2
  exit 5
fi
run_stage "09_extract_swin_fod_features" "$PYTHON" code/multimodalAD/Swin_FOD/extract_features_local.py --checkpoint "$SWIN_CKPT" --manifest "$ROOT_DIR/data/official_inputs/swin_fod_local.csv" --output-dir "$ROOT_DIR/features/fod" --device "$DEVICE"

run_stage "10_train_albef" bash scripts/train_albef.sh "$ALBEF_PRETRAIN_CHECKPOINT" "$ROOT_DIR/checkpoints/albef"

ALBEF_CKPT="$(find "$ROOT_DIR/checkpoints/albef" -name 'checkpoint_*.pth' | sort | tail -1)"
if [ -z "${ALBEF_CKPT:-}" ]; then
  echo "ERROR: ALBEF checkpoint was not produced." >&2
  exit 6
fi
run_stage "11_extract_albef_features" bash scripts/extract_albef_features.sh "$ALBEF_CKPT" "$ROOT_DIR/features/mri_pet" code/multimodalAD/ALBEF/configs/train_ADNI_local.yaml

run_stage "12_dhnn_proto_fusion" "$PYTHON" code/05_train_fusion_model.py --fod-features "$ROOT_DIR/features/fod" --mri-pet-features "$ROOT_DIR/features/mri_pet" --participants "$BIDS_DIR/participants.tsv" --splits-dir "$SPLITS_DIR" --out-dir "$ROOT_DIR/results" --device "$DEVICE"

run_stage "13_evaluate" "$PYTHON" code/05_evaluate.py

run_stage "14_validate_trained_outputs" "$PYTHON" code/10_validate_exact_pipeline.py --root "$ROOT_DIR" --require-trained

echo
echo "Exact pipeline complete."
