#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

INPUT_DIR="${1:-${INPUT_DIR:-$REPO_ROOT/data/Validation-Public}}"
RUN_DIR="${2:-${RUN_DIR:-$REPO_ROOT/runs/predict_v2}}"

NNUNET_BASE="${NNUNET_BASE:-$REPO_ROOT/nnunet_work}"
export nnUNet_raw="${nnUNet_raw:-$NNUNET_BASE/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$NNUNET_BASE/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$NNUNET_BASE/nnUNet_results}"

QUADRANT_DATASET_ID="${QUADRANT_DATASET_ID:-313}"
TOOTH_DATASET_ID="${TOOTH_DATASET_ID:-312}"
QUADRANT_CONFIG="${QUADRANT_CONFIG:-3d_lowres}"
TOOTH_CONFIG="${TOOTH_CONFIG:-3d_fullres}"
FOLD="${FOLD:-all}"
QUADRANT_CHECKPOINT="${QUADRANT_CHECKPOINT:-checkpoint_final.pth}"
TOOTH_CHECKPOINT="${TOOTH_CHECKPOINT:-checkpoint_final.pth}"
OUTPUT_LABEL_SCHEME="${OUTPUT_LABEL_SCHEME:-fdi}"
DISABLE_TTA="${DISABLE_TTA:-1}"

NNUNET_INPUTS_DIR="$RUN_DIR/nnunet_inputs"
QUADRANT_PRED_DIR="$RUN_DIR/quadrant_predictions"
RESIZER_DIR="$RUN_DIR/quadrant_resizer"
CROPPED_INPUTS_DIR="$RUN_DIR/quadrant_cropped_inputs"
TOOTH_PRED_DIR="$RUN_DIR/tooth_predictions"
FINAL_DIR="$RUN_DIR/final"

mkdir -p "$RUN_DIR" "$FINAL_DIR"

python process/prepare_inference_inputs.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$NNUNET_INPUTS_DIR" \
  --overwrite

quadrant_predict_args=(
  -i "$NNUNET_INPUTS_DIR"
  -o "$QUADRANT_PRED_DIR"
  -d "$QUADRANT_DATASET_ID"
  -c "$QUADRANT_CONFIG"
  -f "$FOLD"
  -chk "$QUADRANT_CHECKPOINT"
)
if [[ "$DISABLE_TTA" == "1" ]]; then
  quadrant_predict_args+=(--disable_tta)
fi
nnUNetv2_predict "${quadrant_predict_args[@]}"

python pipeline/postprocess_small_components.py "$QUADRANT_PRED_DIR" \
  --threshold-mm3 50 \
  --in-place

python pipeline/quadrant_locate.py \
  "$NNUNET_INPUTS_DIR" \
  "$QUADRANT_PRED_DIR" \
  "$RESIZER_DIR" \
  "$CROPPED_INPUTS_DIR" \
  --overwrite

tooth_predict_args=(
  -i "$CROPPED_INPUTS_DIR"
  -o "$TOOTH_PRED_DIR"
  -d "$TOOTH_DATASET_ID"
  -c "$TOOTH_CONFIG"
  -f "$FOLD"
  -chk "$TOOTH_CHECKPOINT"
)
if [[ "$DISABLE_TTA" == "1" ]]; then
  tooth_predict_args+=(--disable_tta)
fi
nnUNetv2_predict "${tooth_predict_args[@]}"

python pipeline/quadrant_merge.py \
  "$NNUNET_INPUTS_DIR" \
  "$TOOTH_PRED_DIR" \
  "$RESIZER_DIR" \
  "$FINAL_DIR" \
  --output-label-scheme "$OUTPUT_LABEL_SCHEME"

python pipeline/postprocess_small_components.py "$FINAL_DIR" \
  --threshold-mm3 30 \
  --in-place

echo "[DONE] Final masks: $FINAL_DIR"
