#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python3}"
CHECKPOINT=${1:-}
OUTDIR=${2:-$ROOT_DIR/checkpoints/albef}
CONFIG="${ALBEF_CONFIG:-code/multimodalAD/ALBEF/configs/train_ADNI_local.yaml}"
DEVICE="${DEVICE:-cuda}"
NGPUS=${NGPUS:-1}

mkdir -p "$OUTDIR"

if [ -z "$CHECKPOINT" ]; then
  echo "Usage: $0 <pretrained_checkpoint.pth> [output_dir]" >&2
  exit 1
fi

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "ERROR: Python not found: $PYTHON" >&2
  exit 1
fi

echo "Training ALBEF with checkpoint $CHECKPOINT -> $OUTDIR (ngpus=$NGPUS)"
if [ ! -f "code/multimodalAD/ALBEF/models/model_pretrain3D.py" ]; then
  echo "ERROR: the custom 3-D ALBEF model module is not included in the public source:" >&2
  echo "  code/multimodalAD/ALBEF/models/model_pretrain3D.py" >&2
  echo "The available ALBEF source is retained, but this private/custom module and checkpoint" >&2
  echo "must be supplied before raw MRI/PET extraction can run." >&2
  echo "After supplying it, rerun this command with the compatible checkpoint." >&2
  exit 2
fi

# Use torchrun if multiple gpus; otherwise run single-process
if [ "$NGPUS" -gt 1 ]; then
  torchrun --nproc_per_node=$NGPUS code/multimodalAD/ALBEF/train_ADNI.py --config "$CONFIG" --checkpoint "$CHECKPOINT" --output_dir "$OUTDIR" --device "$DEVICE" --distributed | tee "$OUTDIR/train.log"
else
  "$PYTHON" code/multimodalAD/ALBEF/train_ADNI.py --config "$CONFIG" --checkpoint "$CHECKPOINT" --output_dir "$OUTDIR" --device "$DEVICE" | tee "$OUTDIR/train.log"
fi

echo "ALBEF training wrapper finished"
