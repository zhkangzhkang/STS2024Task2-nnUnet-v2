#!/usr/bin/env python3
"""Select stable pseudo labels from multiple prediction folders.

For semi-supervised training we usually do not have ground truth for unlabeled
volumes. This script scores a pseudo label by the mean pairwise multi-class Dice
between predictions from different checkpoints/epochs, then copies the selected
labels into an output directory.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from itertools import combinations
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def strip_nii_suffix(path: Path | str) -> str:
    name = Path(path).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def load_label(path: Path) -> np.ndarray:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    if not np.issubdtype(array.dtype, np.integer):
        array = np.rint(array).astype(np.int32)
    return array


def dice_binary(a: np.ndarray, b: np.ndarray) -> float:
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return float("nan")
    return float(2 * np.logical_and(a, b).sum() / denom)


def multiclass_dice(a: np.ndarray, b: np.ndarray) -> float:
    labels = sorted((set(np.unique(a).astype(int)) | set(np.unique(b).astype(int))) - {0})
    scores = []
    for label in labels:
        score = dice_binary(a == label, b == label)
        if not np.isnan(score):
            scores.append(score)
    if not scores:
        return 0.0
    return float(np.mean(scores))


def collect_predictions(prediction_dirs: list[Path]) -> dict[str, dict[Path, Path]]:
    cases: dict[str, dict[Path, Path]] = {}
    for pred_dir in prediction_dirs:
        if not pred_dir.exists():
            raise FileNotFoundError(f"Prediction directory does not exist: {pred_dir}")
        for pred in sorted(pred_dir.glob("*.nii.gz")):
            case_id = strip_nii_suffix(pred)
            if case_id.endswith("_Mask"):
                case_id = case_id[:-5]
            cases.setdefault(case_id, {})[pred_dir] = pred
    return cases


def score_case(case_id: str, paths_by_dir: dict[Path, Path], prediction_dirs: list[Path]) -> tuple[float, str]:
    missing = [str(d) for d in prediction_dirs if d not in paths_by_dir]
    if missing:
        return 0.0, f"missing predictions in {missing}"

    arrays = []
    for pred_dir in prediction_dirs:
        arrays.append(load_label(paths_by_dir[pred_dir]))

    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        return 0.0, f"shape mismatch: {sorted(shapes)}"

    pair_scores = [multiclass_dice(a, b) for a, b in combinations(arrays, 2)]
    if not pair_scores:
        return 0.0, "need at least two prediction directories"
    return float(np.mean(pair_scores)), "ok"


def select_pseudo_labels(
    prediction_dirs: list[Path],
    output_dir: Path,
    threshold: float,
    top_k: int | None,
    copy_from: int,
    report_csv: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = collect_predictions(prediction_dirs)
    rows: list[dict[str, str | float | int]] = []

    for case_id, paths_by_dir in sorted(cases.items()):
        score, status = score_case(case_id, paths_by_dir, prediction_dirs)
        rows.append({"case_id": case_id, "score": score, "status": status})
        print(f"[INFO] {case_id}: consistency_dice={score:.4f} ({status})")

    valid_rows = [row for row in rows if row["status"] == "ok" and float(row["score"]) >= threshold]
    valid_rows.sort(key=lambda row: float(row["score"]), reverse=True)
    if top_k is not None:
        valid_rows = valid_rows[:top_k]

    source_dir = prediction_dirs[copy_from]
    for row in valid_rows:
        case_id = str(row["case_id"])
        source = collect_predictions([source_dir])[case_id][source_dir]
        destination = output_dir / f"{case_id}_Mask.nii.gz"
        shutil.copy2(source, destination)
        print(f"[SELECT] {case_id}: copied {source.name} -> {destination.name}")

    report_csv.parent.mkdir(parents=True, exist_ok=True)
    with report_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "score", "status", "selected"])
        writer.writeheader()
        selected = {str(row["case_id"]) for row in valid_rows}
        for row in rows:
            writer.writerow({**row, "selected": str(row["case_id"]) in selected})

    print(f"[DONE] selected {len(valid_rows)} pseudo labels into {output_dir}")
    print(f"[DONE] wrote report: {report_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--copy-from",
        type=int,
        default=-1,
        help="Index of prediction-dirs to copy selected labels from. Default: last directory.",
    )
    parser.add_argument("--report-csv", type=Path, default=Path("pseudo_selection.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.prediction_dirs) < 2:
        raise ValueError("At least two --prediction-dirs are required for consistency scoring.")
    copy_from = args.copy_from
    if copy_from < 0:
        copy_from = len(args.prediction_dirs) + copy_from
    if copy_from < 0 or copy_from >= len(args.prediction_dirs):
        raise IndexError("--copy-from is outside the prediction-dirs range.")
    select_pseudo_labels(
        args.prediction_dirs,
        args.output_dir,
        args.threshold,
        args.top_k,
        copy_from,
        args.report_csv,
    )


if __name__ == "__main__":
    main()
