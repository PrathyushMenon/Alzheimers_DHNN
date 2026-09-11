#!/usr/bin/env bash
set -euo pipefail

BIDS_DIR="${BIDS_DIR:-/mnt/d/ALZ/data/bids}"
CAPS_DIR="${CAPS_DIR:-/mnt/d/ALZ/data/caps}"
OUT_DIR="${OUT_DIR:-/mnt/d/ALZ/data/fod}"

mkdir -p "$OUT_DIR"

if ! command -v dwidenoise >/dev/null 2>&1; then
  echo "MRtrix3 is not on PATH; install MRtrix3 before computing FOD." >&2
  exit 2
fi
if ! command -v dwifslpreproc >/dev/null 2>&1; then
  echo "MRtrix3 dwifslpreproc is not on PATH; exact DWI preprocessing requires it." >&2
  exit 2
fi
if ! command -v mrtransform >/dev/null 2>&1; then
  echo "MRtrix3 mrtransform is not on PATH; exact FOD MNI reorientation requires it." >&2
  exit 2
fi

count=0
failed=0
shopt -s nullglob
for DWI_FILE in "$BIDS_DIR"/sub-*/ses-*/dwi/*_dwi.nii.gz; do
  SUB="$(echo "$DWI_FILE" | grep -oP 'sub-[^/]+')"
  SES="$(echo "$DWI_FILE" | grep -oP 'ses-[^/]+')"
  BVAL="${DWI_FILE%.nii.gz}.bval"
  BVEC="${DWI_FILE%.nii.gz}.bvec"

  if [[ ! -f "$BVAL" || ! -f "$BVEC" ]]; then
    echo "ERROR: missing bval/bvec for $SUB $SES" >&2
    failed=$((failed + 1))
    continue
  fi

  NGRAD="$(awk '{print NF; exit}' "$BVAL")"
  if [[ "${NGRAD:-0}" -lt "${MIN_DWI_VOLUMES:-30}" ]]; then
    echo "ERROR: $SUB $SES has only $NGRAD DWI volumes; exact CSD/FOD needs a full DWI acquisition." >&2
    failed=$((failed + 1))
    continue
  fi

  OUTD="$OUT_DIR/${SUB}_${SES}"
  mkdir -p "$OUTD"
  echo "Processing $SUB $SES"

  dwidenoise "$DWI_FILE" "$OUTD/dwi_denoised.nii.gz" -fslgrad "$BVEC" "$BVAL" -force
  dwifslpreproc "$OUTD/dwi_denoised.nii.gz" "$OUTD/dwi_preproc.nii.gz" -pe_dir AP -rpe_none -fslgrad "$BVEC" "$BVAL" -eddy_options " --slm=linear" -force
  dwibiascorrect ants "$OUTD/dwi_preproc.nii.gz" "$OUTD/dwi_biascorr.nii.gz" -fslgrad "$BVEC" "$BVAL" -force
  dwi2response tournier "$OUTD/dwi_biascorr.nii.gz" "$OUTD/response.txt" -fslgrad "$BVEC" "$BVAL" -force
  dwi2fod csd "$OUTD/dwi_biascorr.nii.gz" "$OUTD/response.txt" "$OUTD/fod_lmax4.nii.gz" -lmax 4 -fslgrad "$BVEC" "$BVAL" -force

  T1W="$(find "$CAPS_DIR/subjects/$SUB/$SES/t1_linear" -name '*_T1w.nii.gz' 2>/dev/null | head -1 || true)"
  AFFINE="$(find "$CAPS_DIR/subjects/$SUB/$SES/t1_linear" -name '*_affine.mat' 2>/dev/null | head -1 || true)"
  if [[ -n "$T1W" && -n "$AFFINE" ]]; then
    mrtransform "$OUTD/fod_lmax4.nii.gz" -linear "$AFFINE" -template "$T1W" -reorient_fod yes "$OUTD/fod_lmax4_MNI.nii.gz" -force
  else
    echo "ERROR: no ClinicaDL T1w affine found for $SUB $SES; refusing unregistered FOD fallback." >&2
    failed=$((failed + 1))
    continue
  fi
  count=$((count + 1))
done

if [[ "$failed" -gt 0 ]]; then
  echo "FOD computation incomplete. Completed subjects: $count; failed subjects: $failed" >&2
  exit 1
fi

echo "FOD computation complete. Completed subjects: $count"
