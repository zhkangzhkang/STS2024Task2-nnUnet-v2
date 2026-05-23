#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
UNLABELED_DIR="${UNLABELED_DIR:-$DATA_ROOT/Train-Unlabeled}"
PSEUDO_WORK_DIR="${PSEUDO_WORK_DIR:-$REPO_ROOT/runs/pseudo_iter1}"

NNUNET_BASE="${NNUNET_BASE:-$REPO_ROOT/nnunet_work}"
export nnUNet_raw="${nnUNet_raw:-$NNUNET_BASE/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$NNUNET_BASE/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$NNUNET_BASE/nnUNet_results}"

TEACHER_QUADRANT_DATASET_ID="${TEACHER_QUADRANT_DATASET_ID:-313}"
TEACHER_TOOTH_DATASET_ID="${TEACHER_TOOTH_DATASET_ID:-312}"
STUDENT_QUADRANT_DATASET_ID="${STUDENT_QUADRANT_DATASET_ID:-323}"
STUDENT_TOOTH_DATASET_ID="${STUDENT_TOOTH_DATASET_ID:-322}"

QUADRANT_CONFIG="${QUADRANT_CONFIG:-3d_lowres}"
TOOTH_CONFIG="${TOOTH_CONFIG:-3d_fullres}"
FOLD="${FOLD:-all}"
CHECKPOINTS="${CHECKPOINTS:-checkpoint_best.pth checkpoint_final.pth}"
LABEL_SCHEME="${LABEL_SCHEME:-sequential}"
PSEUDO_OUTPUT_LABEL_SCHEME="${PSEUDO_OUTPUT_LABEL_SCHEME:-$LABEL_SCHEME}"
PSEUDO_THRESHOLD="${PSEUDO_THRESHOLD:-0.90}"
PSEUDO_TOP_K="${PSEUDO_TOP_K:-30}"

mkdir -p "$PSEUDO_WORK_DIR"

prediction_dirs=()
for checkpoint in $CHECKPOINTS; do
  safe_checkpoint="${checkpoint%.pth}"
  safe_checkpoint="${safe_checkpoint//[^A-Za-z0-9_]/_}"
  run_dir="$PSEUDO_WORK_DIR/predict_${safe_checkpoint}"
  echo "[INFO] Predicting unlabeled data with $checkpoint"
  QUADRANT_DATASET_ID="$TEACHER_QUADRANT_DATASET_ID" \
  TOOTH_DATASET_ID="$TEACHER_TOOTH_DATASET_ID" \
  QUADRANT_CHECKPOINT="$checkpoint" \
  TOOTH_CHECKPOINT="$checkpoint" \
  OUTPUT_LABEL_SCHEME="$PSEUDO_OUTPUT_LABEL_SCHEME" \
  QUADRANT_CONFIG="$QUADRANT_CONFIG" \
  TOOTH_CONFIG="$TOOTH_CONFIG" \
  FOLD="$FOLD" \
  bash scripts/predict_v2.sh "$UNLABELED_DIR" "$run_dir"
  prediction_dirs+=("$run_dir/final")
done

selected_dir="$PSEUDO_WORK_DIR/selected_pseudo_labels"
report_csv="$PSEUDO_WORK_DIR/pseudo_selection.csv"
python process/select_pseudo_dice.py \
  --prediction-dirs "${prediction_dirs[@]}" \
  --output-dir "$selected_dir" \
  --threshold "$PSEUDO_THRESHOLD" \
  --top-k "$PSEUDO_TOP_K" \
  --report-csv "$report_csv"

python process/prepare_nnunetv2_datasets.py \
  --data-root "$DATA_ROOT" \
  --nnunet-raw "$nnUNet_raw" \
  --quadrant-dataset-id "$STUDENT_QUADRANT_DATASET_ID" \
  --tooth-dataset-id "$STUDENT_TOOTH_DATASET_ID" \
  --label-scheme "$LABEL_SCHEME" \
  --pseudo-image-dir "$UNLABELED_DIR" \
  --pseudo-label-dir "$selected_dir" \
  --overwrite

nnUNetv2_plan_and_preprocess -d "$STUDENT_QUADRANT_DATASET_ID" --verify_dataset_integrity
nnUNetv2_train "$STUDENT_QUADRANT_DATASET_ID" "$QUADRANT_CONFIG" "$FOLD" --npz

nnUNetv2_plan_and_preprocess -d "$STUDENT_TOOTH_DATASET_ID" --verify_dataset_integrity
nnUNetv2_train "$STUDENT_TOOTH_DATASET_ID" "$TOOTH_CONFIG" "$FOLD" --npz
