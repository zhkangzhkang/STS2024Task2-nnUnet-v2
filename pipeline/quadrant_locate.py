#!/usr/bin/env python3
"""Crop each predicted tooth quadrant into nnU-Net v2 inference inputs."""

from __future__ import annotations

import argparse
import shutil
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


def find_image(image_dir: Path, case_id: str) -> Path:
    candidates = [
        image_dir / f"{case_id}_0000.nii.gz",
        image_dir / f"{case_id}.nii.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No source image found for case {case_id} in {image_dir}")


def crop_origin_from_array_index(image: sitk.Image, bbox_min_zyx: np.ndarray) -> tuple[float, float, float]:
    z_min, y_min, x_min = [int(v) for v in bbox_min_zyx]
    return image.TransformIndexToPhysicalPoint((x_min, y_min, z_min))


def crop_quadrants(
    image_dir: Path,
    quadrant_dir: Path,
    resizer_dir: Path,
    crop_dir: Path,
    padding: int = 2,
    overwrite: bool = False,
    exclude_top_fraction: float = 0.15,
    exclude_y_after_fraction: float | None = 0.80,
    y_exclusion_min_size: int = 500,
) -> None:
    if overwrite:
        if resizer_dir.exists():
            shutil.rmtree(resizer_dir)
        if crop_dir.exists():
            shutil.rmtree(crop_dir)
    resizer_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    for quadrant_path in sorted(quadrant_dir.glob("*.nii.gz")):
        case_id = strip_nii_suffix(quadrant_path)
        image_path = find_image(image_dir, case_id)
        image = sitk.ReadImage(str(image_path))
        image_array = sitk.GetArrayFromImage(image)

        quadrant_mask = sitk.ReadImage(str(quadrant_path))
        quadrant_array = sitk.GetArrayFromImage(quadrant_mask)
        if not np.issubdtype(quadrant_array.dtype, np.integer):
            quadrant_array = np.rint(quadrant_array).astype(np.int16)

        if exclude_top_fraction > 0:
            z_cut = int(quadrant_array.shape[0] * exclude_top_fraction)
            quadrant_array[:z_cut] = 0
        if exclude_y_after_fraction is not None and quadrant_array.shape[1] > y_exclusion_min_size:
            y_cut = int(quadrant_array.shape[1] * exclude_y_after_fraction)
            quadrant_array[:, y_cut:, :] = 0

        resizer_dict: dict[str, tuple[slice, slice, slice]] = {}
        for quadrant in range(1, 5):
            coords = np.where(quadrant_array == quadrant)
            if len(coords[0]) == 0:
                print(f"[WARN] {case_id}: no voxels found for quadrant {quadrant}; skipped")
                continue

            bbox_min = np.maximum(np.min(coords, axis=1) - padding, 0)
            bbox_max = np.minimum(np.max(coords, axis=1) + padding + 1, np.array(quadrant_array.shape))
            slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(bbox_min, bbox_max))

            crop_array = image_array[slices]
            crop_image = sitk.GetImageFromArray(crop_array)
            crop_image.SetSpacing(image.GetSpacing())
            crop_image.SetDirection(image.GetDirection())
            crop_image.SetOrigin(crop_origin_from_array_index(image, bbox_min))

            crop_name = f"{case_id}_quadrant_{quadrant}_0000.nii.gz"
            sitk.WriteImage(crop_image, str(crop_dir / crop_name))
            resizer_dict[crop_name] = slices
            print(f"[INFO] {crop_name}: {slices}")

        np.save(resizer_dir / f"{case_id}.npy", resizer_dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path, help="nnU-Net-style input image directory.")
    parser.add_argument("quadrant_dir", type=Path, help="Predicted quadrant masks from nnUNetv2_predict.")
    parser.add_argument("resizer_dir", type=Path, help="Where crop slice metadata will be saved.")
    parser.add_argument("crop_dir", type=Path, help="Where cropped quadrant images will be written.")
    parser.add_argument("--padding", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--exclude-top-fraction", type=float, default=0.15)
    parser.add_argument("--exclude-y-after-fraction", type=float, default=0.80)
    parser.add_argument("--disable-y-exclusion", action="store_true")
    parser.add_argument("--y-exclusion-min-size", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    crop_quadrants(
        args.image_dir,
        args.quadrant_dir,
        args.resizer_dir,
        args.crop_dir,
        padding=args.padding,
        overwrite=args.overwrite,
        exclude_top_fraction=args.exclude_top_fraction,
        exclude_y_after_fraction=None if args.disable_y_exclusion else args.exclude_y_after_fraction,
        y_exclusion_min_size=args.y_exclusion_min_size,
    )


if __name__ == "__main__":
    main()
