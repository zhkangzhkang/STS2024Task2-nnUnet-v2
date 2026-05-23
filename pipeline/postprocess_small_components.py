#!/usr/bin/env python3
"""Remove small connected components from NIfTI label maps."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def remove_small_components(mask: sitk.Image, threshold_mm3: float) -> sitk.Image:
    array = sitk.GetArrayFromImage(mask)
    if not np.issubdtype(array.dtype, np.integer):
        array = np.rint(array).astype(np.int32)

    voxel_volume = float(np.prod(mask.GetSpacing()))
    min_size_voxels = max(1, int(round(threshold_mm3 / voxel_volume)))
    cleaned = np.zeros(array.shape, dtype=array.dtype)

    for label_value in np.unique(array):
        label_value = int(label_value)
        if label_value == 0:
            continue
        binary = sitk.GetImageFromArray((array == label_value).astype(np.uint8))
        binary.CopyInformation(mask)
        components = sitk.ConnectedComponent(binary)
        relabeled = sitk.RelabelComponent(components, minimumObjectSize=min_size_voxels)
        kept = sitk.GetArrayFromImage(relabeled) > 0
        cleaned[kept] = label_value

    out = sitk.GetImageFromArray(cleaned)
    out.CopyInformation(mask)
    return out


def process_dir(input_dir: Path, output_dir: Path, threshold_mm3: float, overwrite: bool) -> None:
    if output_dir.exists() and output_dir != input_dir and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.nii.gz"))
    if not files:
        raise RuntimeError(f"No .nii.gz masks found in {input_dir}")

    for mask_path in files:
        mask = sitk.ReadImage(str(mask_path))
        cleaned = remove_small_components(mask, threshold_mm3)
        dst = output_dir / mask_path.name
        sitk.WriteImage(cleaned, str(dst))
        print(f"[DONE] {mask_path.name}: removed components smaller than {threshold_mm3} mm^3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threshold-mm3", type=float, default=30.0)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.in_place:
        output_dir = args.input_dir
    elif args.output_dir is not None:
        output_dir = args.output_dir
    else:
        raise ValueError("Pass --in-place or --output-dir.")
    process_dir(args.input_dir, output_dir, args.threshold_mm3, args.overwrite)


if __name__ == "__main__":
    main()
