#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
NNUNET_BASE="${NNUNET_BASE:-$REPO_ROOT/nnunet_work}"
export nnUNet_raw="${nnUNet_raw:-$NNUNET_BASE/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$NNUNET_BASE/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$NNUNET_BASE/nnUNet_results}"

QUADRANT_DATASET_ID="${QUADRANT_DATASET_ID:-313}"
TOOTH_DATASET_ID="${TOOTH_DATASET_ID:-312}"
LABEL_SCHEME="${LABEL_SCHEME:-auto}"
FOLD="${FOLD:-all}"
QUADRANT_CONFIG="${QUADRANT_CONFIG:-3d_lowres}"
TOOTH_CONFIG="${TOOTH_CONFIG:-3d_fullres}"

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

python process/prepare_nnunetv2_datasets.py \
  --data-root "$DATA_ROOT" \
  --nnunet-raw "$nnUNet_raw" \
  --quadrant-dataset-id "$QUADRANT_DATASET_ID" \
  --tooth-dataset-id "$TOOTH_DATASET_ID" \
  --label-scheme "$LABEL_SCHEME" \
  --overwrite

nnUNetv2_plan_and_preprocess -d "$QUADRANT_DATASET_ID" --verify_dataset_integrity
nnUNetv2_train "$QUADRANT_DATASET_ID" "$QUADRANT_CONFIG" "$FOLD" --npz

nnUNetv2_plan_and_preprocess -d "$TOOTH_DATASET_ID" --verify_dataset_integrity
nnUNetv2_train "$TOOTH_DATASET_ID" "$TOOTH_CONFIG" "$FOLD" --npz
