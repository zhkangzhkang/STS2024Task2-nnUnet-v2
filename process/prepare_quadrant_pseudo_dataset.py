#!/usr/bin/env python3
"""Build a quadrant-only nnU-Net v2 dataset with manually selected pseudo labels.

Use this when pseudo labels are already quadrant masks with labels 0..4. This
script combines original labeled data and selected unlabeled quadrant labels to
train a better first-stage quadrant model.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from prepare_nnunetv2_datasets import (
    CasePair,
    collect_case_pairs,
    dataset_dir,
    detect_label_scheme,
    normalize_case_id,
    read_mask_array,
    remap_to_quadrants,
    strip_nii_suffix,
    write_dataset_json,
    write_like,
)


def make_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Pass --overwrite to recreate it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_image(src: Path, images_dir: Path, case_id: str) -> None:
    shutil.copy2(src, images_dir / f"{case_id}_0000.nii.gz")


def find_pseudo_image(image_dir: Path, case_id: str) -> Path | None:
    candidates = [
        image_dir / f"{case_id}.nii.gz",
        image_dir / f"{case_id}_0000.nii.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(image_dir.glob(f"{case_id}*.nii.gz"))
    if matches:
        return matches[0]
    return None


def collect_manual_pseudo_pairs(image_dir: Path, label_dir: Path) -> list[CasePair]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Pseudo image directory does not exist: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Pseudo label directory does not exist: {label_dir}")

    pairs: list[CasePair] = []
    for label in sorted(label_dir.glob("*.nii.gz")):
        case_id = normalize_case_id(strip_nii_suffix(label))
        image = find_pseudo_image(image_dir, case_id)
        if image is None:
            raise FileNotFoundError(f"No pseudo source image found for label {label.name} in {image_dir}")
        pairs.append(CasePair(case_id=case_id, image=image, mask=label, source="manual_pseudo"))

    if not pairs:
        raise RuntimeError(f"No .nii.gz pseudo labels found in {label_dir}")
    return pairs


def validate_quadrant_mask(case_id: str, mask_array: np.ndarray) -> np.ndarray:
    if not np.issubdtype(mask_array.dtype, np.integer):
        mask_array = np.rint(mask_array).astype(np.int16)
    unique = sorted(int(v) for v in np.unique(mask_array))
    invalid = [v for v in unique if v < 0 or v > 4]
    if invalid:
        raise ValueError(
            f"{case_id} is not a quadrant mask. Expected labels 0..4, "
            f"got labels {unique[:30]}"
        )
    return mask_array.astype(np.uint8)


def build_dataset(
    data_root: Path,
    raw_root: Path,
    dataset_id: int,
    dataset_name: str,
    label_scheme: str,
    pseudo_image_dir: Path,
    pseudo_label_dir: Path,
    overwrite: bool,
) -> None:
    labeled_image_dir = data_root / "Train-Labeled" / "Images"
    labeled_mask_dir = data_root / "Train-Labeled" / "Masks"
    labeled_cases = collect_case_pairs(labeled_image_dir, labeled_mask_dir, source="labeled")

    if label_scheme == "auto":
        label_scheme = detect_label_scheme(case.mask for case in labeled_cases)
    print(f"[INFO] Labeled source label_scheme={label_scheme}")

    pseudo_cases = collect_manual_pseudo_pairs(pseudo_image_dir, pseudo_label_dir)
    print(f"[INFO] Found {len(labeled_cases)} labeled cases")
    print(f"[INFO] Found {len(pseudo_cases)} manually selected quadrant pseudo labels")

    out_dir = dataset_dir(raw_root, dataset_id, dataset_name)
    make_clean_dir(out_dir, overwrite)
    images_tr = out_dir / "imagesTr"
    labels_tr = out_dir / "labelsTr"
    images_tr.mkdir()
    labels_tr.mkdir()

    for case in labeled_cases:
        image = sitk.ReadImage(str(case.image))
        image_array = sitk.GetArrayFromImage(image)
        _, mask_array = read_mask_array(case.mask)
        if mask_array.shape != image_array.shape:
            raise ValueError(
                f"Image/mask shape mismatch for {case.case_id}: "
                f"image={image_array.shape}, mask={mask_array.shape}"
            )
        copy_image(case.image, images_tr, case.case_id)
        quadrant_array = remap_to_quadrants(mask_array, label_scheme)
        write_like(image, quadrant_array, labels_tr / f"{case.case_id}.nii.gz")

    for case in pseudo_cases:
        image = sitk.ReadImage(str(case.image))
        image_array = sitk.GetArrayFromImage(image)
        _, pseudo_array = read_mask_array(case.mask)
        if pseudo_array.shape != image_array.shape:
            raise ValueError(
                f"Image/pseudo-label shape mismatch for {case.case_id}: "
                f"image={image_array.shape}, pseudo={pseudo_array.shape}"
            )
        copy_image(case.image, images_tr, case.case_id)
        quadrant_array = validate_quadrant_mask(case.case_id, pseudo_array)
        write_like(image, quadrant_array, labels_tr / f"{case.case_id}.nii.gz")

    write_dataset_json(
        out_dir,
        labels={
            "background": 0,
            "quadrant_1": 1,
            "quadrant_2": 2,
            "quadrant_3": 3,
            "quadrant_4": 4,
        },
        num_training=len(labeled_cases) + len(pseudo_cases),
        description=(
            "STS 2024 quadrant segmentation with labeled data and manually "
            "selected unlabeled quadrant pseudo labels."
        ),
    )

    print(f"[DONE] Wrote dataset: {out_dir}")
    print(f"[DONE] numTraining={len(labeled_cases) + len(pseudo_cases)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--nnunet-raw",
        type=Path,
        default=Path(os.environ.get("nnUNet_raw", "nnUNet_raw")),
    )
    parser.add_argument("--dataset-id", type=int, default=323)
    parser.add_argument("--dataset-name", default="STS2024_ToothQuadrantsManualPseudo")
    parser.add_argument("--label-scheme", choices=["auto", "sequential", "fdi"], default="auto")
    parser.add_argument("--pseudo-image-dir", type=Path, default=None)
    parser.add_argument("--pseudo-label-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pseudo_image_dir = args.pseudo_image_dir or (args.data_root / "Train-Unlabeled")
    build_dataset(
        args.data_root,
        args.nnunet_raw,
        args.dataset_id,
        args.dataset_name,
        args.label_scheme,
        pseudo_image_dir,
        args.pseudo_label_dir,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
