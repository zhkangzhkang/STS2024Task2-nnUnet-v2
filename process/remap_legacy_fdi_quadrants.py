#!/usr/bin/env python3
"""Remap legacy swapped FDI quadrant labels to standard FDI labels.

Older predictions produced by this repository may have used:
  11-18 <-> 21-28 and 31-38 <-> 41-48
This script swaps them back while preserving image metadata.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def remap_array(array: np.ndarray) -> np.ndarray:
    out = np.zeros(array.shape, dtype=np.uint16)
    out[array == 0] = 0

    for tooth in range(1, 9):
        out[array == 10 + tooth] = 20 + tooth
        out[array == 20 + tooth] = 10 + tooth
        out[array == 30 + tooth] = 40 + tooth
        out[array == 40 + tooth] = 30 + tooth

    known = {0}
    for base in (10, 20, 30, 40):
        known.update(range(base + 1, base + 9))
    unknown = sorted(set(np.unique(array).astype(int)) - known)
    if unknown:
        print(f"[WARN] unknown labels left as 0: {unknown}")
    return out


def remap_file(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} exists. Pass --overwrite to replace it.")
    image = sitk.ReadImage(str(src))
    array = sitk.GetArrayFromImage(image)
    if not np.issubdtype(array.dtype, np.integer):
        array = np.rint(array).astype(np.int32)
    remapped = remap_array(array)
    out = sitk.GetImageFromArray(remapped)
    out.CopyInformation(image)
    sitk.WriteImage(out, str(dst))


def remap_dir(input_dir: Path, output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for src in sorted(input_dir.glob("*.nii.gz")):
        remap_file(src, output_dir / src.name, overwrite)
        count += 1
    if count == 0:
        raise RuntimeError(f"No .nii.gz files found in {input_dir}")
    print(f"[DONE] remapped {count} files into {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    remap_dir(args.input_dir, args.output_dir, args.overwrite)


if __name__ == "__main__":
    main()
