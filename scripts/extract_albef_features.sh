#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CHECKPOINT="${1:?Usage: extract_albef_features.sh <checkpoint> <output_dir> [config]}"
OUTPUT_DIR="${2:?Usage: extract_albef_features.sh <checkpoint> <output_dir> [config]}"
CONFIG="${3:-$ROOT_DIR/code/multimodalAD/ALBEF/configs/train_ADNI_local.yaml}"
ALBEF_DIR="$ROOT_DIR/code/multimodalAD/ALBEF"
RUN_DIR="$OUTPUT_DIR/albef_extract_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR"
cd "$ALBEF_DIR"

if [[ ! -f models/model_pretrain3D.py ]]; then
  echo "ERROR: missing custom 3-D ALBEF module: $ALBEF_DIR/models/model_pretrain3D.py" >&2
  exit 2
fi

python test_pretrain3D.py --config "$CONFIG" --checkpoint "$CHECKPOINT" --output_dir "$RUN_DIR"

for split in train val test; do
  case "$split" in
    train) prefix=tr ;;
    val) prefix=val ;;
    test) prefix=test ;;
  esac
  source_file="$RUN_DIR/${prefix}_feat_fusioned.npz"
  [[ -f "$source_file" ]] || { echo "ERROR: missing ALBEF output: $source_file" >&2; exit 3; }
  python - "$source_file" "$OUTPUT_DIR/${split}_features.npy" <<'PY'
import sys
import numpy as np

source, target = sys.argv[1], sys.argv[2]
with np.load(source) as archive:
    values = np.asarray(archive["fusioned"], dtype=np.float32)
if values.ndim != 2 or not np.isfinite(values).all():
    raise SystemExit(f"Invalid ALBEF feature array: {source}")
np.save(target, values)
print(f"{target}: {values.shape}")
PY
done
