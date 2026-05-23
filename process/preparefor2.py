#!/usr/bin/env python3
"""Create cropped quadrant images and labels for the second-stage model.

This compatibility utility writes plain image/label folders. For complete
nnU-Net v2 Dataset folders, use process/prepare_nnunetv2_datasets.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from prepare_nnunetv2_datasets import (
    collect_case_pairs,
    crop_origin_from_array_index,
    quadrant_ranges,
    read_mask_array,
    remap_to_quadrant_teeth,
)


def process_images(
    image_dir: Path,
    mask_dir: Path,
    output_image_dir: Path,
    output_mask_dir: Path,
    label_scheme: str,
    padding: int,
    overwrite: bool,
) -> None:
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    cases = collect_case_pairs(image_dir, mask_dir, source="labeled")

    for case in cases:
        image = sitk.ReadImage(str(case.image))
        mask, mask_array = read_mask_array(case.mask)
        image_array = sitk.GetArrayFromImage(image)

        for quadrant, (start, end) in quadrant_ranges(label_scheme).items():
            coords = np.where((mask_array >= start) & (mask_array <= end))
            if len(coords[0]) == 0:
                print(f"[WARN] {case.case_id}: no voxels found for quadrant {quadrant}; skipped")
                continue

            bbox_min = np.maximum(np.min(coords, axis=1) - padding, 0)
            bbox_max = np.minimum(np.max(coords, axis=1) + padding + 1, np.array(mask_array.shape))
            slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(bbox_min, bbox_max))

            out_case_id = f"{case.case_id}_q{quadrant}"
            image_path = output_image_dir / f"{out_case_id}_0000.nii.gz"
            mask_path = output_mask_dir / f"{out_case_id}.nii.gz"
            if (image_path.exists() or mask_path.exists()) and not overwrite:
                raise FileExistsError(f"{out_case_id} exists. Pass --overwrite to replace it.")

            roi_image = image_array[slices]
            roi_mask = mask_array[slices]
            remapped_mask = remap_to_quadrant_teeth(roi_mask, quadrant, label_scheme)

            new_image = sitk.GetImageFromArray(roi_image)
            new_image.SetSpacing(image.GetSpacing())
            new_image.SetDirection(image.GetDirection())
            new_image.SetOrigin(crop_origin_from_array_index(image, bbox_min))

            new_mask = sitk.GetImageFromArray(remapped_mask)
            new_mask.CopyInformation(new_image)

            sitk.WriteImage(new_image, str(image_path))
            sitk.WriteImage(new_mask, str(mask_path))
            print(f"[DONE] {out_case_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=Path("data/Train-Labeled/Images"))
    parser.add_argument("--mask-dir", type=Path, default=Path("data/Train-Labeled/Masks"))
    parser.add_argument("--output-image-dir", type=Path, required=True)
    parser.add_argument("--output-mask-dir", type=Path, required=True)
    parser.add_argument("--label-scheme", choices=["sequential", "fdi"], default="sequential")
    parser.add_argument("--padding", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_images(
        args.image_dir,
        args.mask_dir,
        args.output_image_dir,
        args.output_mask_dir,
        args.label_scheme,
        args.padding,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
