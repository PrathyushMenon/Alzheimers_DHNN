#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR/code/multimodalAD/Swin_FOD"

PYTHON="${PYTHON:-python3}"
OUTDIR="${1:-$ROOT_DIR/checkpoints/swin_fod}"
MANIFEST="${SWIN_FOD_MANIFEST:-$ROOT_DIR/data/official_inputs/swin_fod_local.csv}"
MAX_EPOCHS="${SWIN_MAX_EPOCHS:-100}"
BATCH_SIZE="${SWIN_BATCH_SIZE:-1}"
WORKERS="${SWIN_WORKERS:-2}"

mkdir -p "$OUTDIR"

if [ ! -f "$MANIFEST" ]; then
  echo "ERROR: Swin-FOD local manifest missing: $MANIFEST" >&2
  echo "Run: python code/11_prepare_official_inputs.py" >&2
  exit 2
fi

echo "Training official Swin-FOD"
echo "manifest=$MANIFEST"
echo "outdir=$OUTDIR"

"$PYTHON" main.py \
  --json_list "$MANIFEST" \
  --modality fod \
  --max_epochs "$MAX_EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --workers "$WORKERS" \
  --feature_size 24 \
  --in_channels 3 \
  --out_channels 3 \
  --roi_x 32 --roi_y 32 --roi_z 32 \
  --optim_lr 1e-4 \
  --lrschedule warmup_cosine \
  --save_checkpoint \
  --logdir "$OUTDIR"
