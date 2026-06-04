#!/usr/bin/env python3
"""Evaluate tooth instance segmentation predictions against NIfTI labels.

The default label set is FDI permanent teeth: 11-18, 21-28, 31-38, 41-48.
Outputs one per-label CSV and one summary CSV with Dice and HD95 in mm.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import SimpleITK as sitk


FDI_LABELS = list(range(11, 19)) + list(range(21, 29)) + list(range(31, 39)) + list(range(41, 49))
QUADRANT_LABELS = [1, 2, 3, 4]


@dataclass(frozen=True)
class CaseMatch:
    case_id: str
    pred: Path
    gt: Path


def strip_nii_suffix(path: Path | str) -> str:
    name = Path(path).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def normalize_case_id(path: Path | str) -> str:
    case_id = strip_nii_suffix(path)
    for suffix in ("_0000", "_Mask", "_mask", "_Label", "_label", "_Segmentation", "_segmentation", "_seg", "_gt", "_GT"):
        if case_id.endswith(suffix):
            case_id = case_id[: -len(suffix)]
    return case_id


def collect_cases(pred_dir: Path, gt_dir: Path) -> list[CaseMatch]:
    preds = {normalize_case_id(p): p for p in sorted(pred_dir.glob("*.nii.gz"))}
    gts = {normalize_case_id(p): p for p in sorted(gt_dir.glob("*.nii.gz"))}

    missing_gt = sorted(set(preds) - set(gts))
    missing_pred = sorted(set(gts) - set(preds))
    if missing_gt:
        print(f"[WARN] {len(missing_gt)} predictions have no GT. First: {missing_gt[:10]}")
    if missing_pred:
        print(f"[WARN] {len(missing_pred)} GT files have no prediction. First: {missing_pred[:10]}")

    common = sorted(set(preds) & set(gts))
    if not common:
        raise RuntimeError(f"No matched .nii.gz cases found between {pred_dir} and {gt_dir}")
    return [CaseMatch(case_id=case_id, pred=preds[case_id], gt=gts[case_id]) for case_id in common]


def read_label(path: Path) -> tuple[sitk.Image, np.ndarray]:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    if not np.issubdtype(array.dtype, np.integer):
        array = np.rint(array).astype(np.int32)
    return image, array.astype(np.int32, copy=False)


def dice_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    denom = pred_sum + gt_sum
    if denom == 0:
        return math.nan
    return float(2 * np.logical_and(pred, gt).sum() / denom)


def crop_union(pred: np.ndarray, gt: np.ndarray, margin: int) -> tuple[np.ndarray, np.ndarray] | None:
    union = pred | gt
    coords = np.argwhere(union)
    if coords.size == 0:
        return None
    lo = np.maximum(coords.min(axis=0) - margin, 0)
    hi = np.minimum(coords.max(axis=0) + margin + 1, np.array(union.shape))
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    return pred[slices], gt[slices]


def hd95_binary(pred: np.ndarray, gt: np.ndarray, spacing_xyz: tuple[float, float, float], margin: int) -> float:
    if not pred.any() and not gt.any():
        return math.nan
    if not pred.any() or not gt.any():
        return math.nan

    cropped = crop_union(pred, gt, margin)
    if cropped is None:
        return math.nan
    pred_crop, gt_crop = cropped

    pred_img = sitk.GetImageFromArray(pred_crop.astype(np.uint8))
    gt_img = sitk.GetImageFromArray(gt_crop.astype(np.uint8))
    pred_img.SetSpacing(spacing_xyz)
    gt_img.SetSpacing(spacing_xyz)

    pred_surface = sitk.LabelContour(pred_img, fullyConnected=True)
    gt_surface = sitk.LabelContour(gt_img, fullyConnected=True)

    pred_surface_arr = sitk.GetArrayViewFromImage(pred_surface).astype(bool)
    gt_surface_arr = sitk.GetArrayViewFromImage(gt_surface).astype(bool)
    if not pred_surface_arr.any() or not gt_surface_arr.any():
        return math.nan

    pred_distance = sitk.Abs(
        sitk.SignedMaurerDistanceMap(pred_img, squaredDistance=False, useImageSpacing=True)
    )
    gt_distance = sitk.Abs(
        sitk.SignedMaurerDistanceMap(gt_img, squaredDistance=False, useImageSpacing=True)
    )

    pred_distance_arr = sitk.GetArrayViewFromImage(pred_distance)
    gt_distance_arr = sitk.GetArrayViewFromImage(gt_distance)
    distances = np.concatenate(
        [
            gt_distance_arr[pred_surface_arr],
            pred_distance_arr[gt_surface_arr],
        ]
    )
    if distances.size == 0:
        return math.nan
    return float(np.percentile(distances, 95))


def labels_from_args(mode: str, values: list[int] | None) -> list[int]:
    if values:
        return values
    if mode == "fdi":
        return FDI_LABELS
    if mode == "quadrant":
        return QUADRANT_LABELS
    raise ValueError(f"Unknown label mode: {mode}")


def status_for(pred_mask: np.ndarray, gt_mask: np.ndarray) -> str:
    pred_any = bool(pred_mask.any())
    gt_any = bool(gt_mask.any())
    if pred_any and gt_any:
        return "both_present"
    if pred_any and not gt_any:
        return "false_positive"
    if gt_any and not pred_any:
        return "missing_prediction"
    return "both_absent"


def safe_mean(values: Iterable[float]) -> float:
    finite = [float(v) for v in values if not math.isnan(float(v)) and math.isfinite(float(v))]
    if not finite:
        return math.nan
    return float(np.mean(finite))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    pred_dir: Path,
    gt_dir: Path,
    labels: list[int],
    output_csv: Path,
    summary_csv: Path,
    crop_margin: int,
) -> None:
    matches = collect_cases(pred_dir, gt_dir)
    rows: list[dict[str, object]] = []
    foreground_rows: list[dict[str, object]] = []

    for match in matches:
        pred_img, pred_arr = read_label(match.pred)
        gt_img, gt_arr = read_label(match.gt)

        if pred_arr.shape != gt_arr.shape:
            raise ValueError(
                f"Shape mismatch for {match.case_id}: pred={pred_arr.shape}, gt={gt_arr.shape}"
            )
        if pred_img.GetSpacing() != gt_img.GetSpacing():
            print(
                f"[WARN] Spacing mismatch for {match.case_id}: "
                f"pred={pred_img.GetSpacing()}, gt={gt_img.GetSpacing()}. Using GT spacing."
            )

        spacing = gt_img.GetSpacing()
        print(f"[INFO] Evaluating {match.case_id}")

        for label in labels:
            pred_mask = pred_arr == label
            gt_mask = gt_arr == label
            status = status_for(pred_mask, gt_mask)
            dice = dice_binary(pred_mask, gt_mask)
            hd95 = hd95_binary(pred_mask, gt_mask, spacing, crop_margin)
            rows.append(
                {
                    "case_id": match.case_id,
                    "label": label,
                    "dice": dice,
                    "hd95_mm": hd95,
                    "pred_voxels": int(pred_mask.sum()),
                    "gt_voxels": int(gt_mask.sum()),
                    "status": status,
                }
            )

        pred_fg = np.isin(pred_arr, labels)
        gt_fg = np.isin(gt_arr, labels)
        foreground_rows.append(
            {
                "case_id": match.case_id,
                "label": "foreground",
                "dice": dice_binary(pred_fg, gt_fg),
                "hd95_mm": hd95_binary(pred_fg, gt_fg, spacing, crop_margin),
                "pred_voxels": int(pred_fg.sum()),
                "gt_voxels": int(gt_fg.sum()),
                "status": status_for(pred_fg, gt_fg),
            }
        )

    write_csv(
        output_csv,
        rows + foreground_rows,
        ["case_id", "label", "dice", "hd95_mm", "pred_voxels", "gt_voxels", "status"],
    )

    summary_rows: list[dict[str, object]] = []
    tooth_rows = [row for row in rows if row["status"] != "both_absent"]
    summary_rows.append(
        {
            "scope": "all_teeth",
            "label": "mean",
            "n": len(tooth_rows),
            "mean_dice": safe_mean(float(row["dice"]) for row in tooth_rows),
            "mean_hd95_mm": safe_mean(float(row["hd95_mm"]) for row in tooth_rows),
        }
    )
    summary_rows.append(
        {
            "scope": "foreground",
            "label": "foreground",
            "n": len(foreground_rows),
            "mean_dice": safe_mean(float(row["dice"]) for row in foreground_rows),
            "mean_hd95_mm": safe_mean(float(row["hd95_mm"]) for row in foreground_rows),
        }
    )
    for label in labels:
        label_rows = [row for row in rows if row["label"] == label and row["status"] != "both_absent"]
        summary_rows.append(
            {
                "scope": "per_label",
                "label": label,
                "n": len(label_rows),
                "mean_dice": safe_mean(float(row["dice"]) for row in label_rows),
                "mean_hd95_mm": safe_mean(float(row["hd95_mm"]) for row in label_rows),
            }
        )

    write_csv(summary_csv, summary_rows, ["scope", "label", "n", "mean_dice", "mean_hd95_mm"])

    print(f"[DONE] cases: {len(matches)}")
    print(f"[DONE] wrote per-case metrics: {output_csv}")
    print(f"[DONE] wrote summary metrics: {summary_csv}")
    print(
        "[RESULT] all_teeth mean Dice="
        f"{summary_rows[0]['mean_dice']:.4f}, mean HD95={summary_rows[0]['mean_hd95_mm']:.4f} mm"
    )
    print(
        "[RESULT] foreground Dice="
        f"{summary_rows[1]['mean_dice']:.4f}, foreground HD95={summary_rows[1]['mean_hd95_mm']:.4f} mm"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=Path("runs/evaluation/per_case_metrics.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("runs/evaluation/summary_metrics.csv"))
    parser.add_argument("--label-mode", choices=["fdi", "quadrant"], default="fdi")
    parser.add_argument("--label-values", type=int, nargs="+", default=None)
    parser.add_argument("--crop-margin", type=int, default=8, help="Voxel margin around each label for HD95.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = labels_from_args(args.label_mode, args.label_values)
    evaluate(args.pred_dir, args.gt_dir, labels, args.output_csv, args.summary_csv, args.crop_margin)


if __name__ == "__main__":
    main()
