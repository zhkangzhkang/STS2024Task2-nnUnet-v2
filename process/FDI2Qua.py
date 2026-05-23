#!/usr/bin/env python3
"""Convert tooth instance labels to quadrant labels.

This is kept as a standalone utility for compatibility. The recommended
nnU-Net v2 workflow calls process/prepare_nnunetv2_datasets.py instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import SimpleITK as sitk

from prepare_nnunetv2_datasets import read_mask_array, remap_to_quadrants, write_like


def convert_directory(input_dir: Path, output_dir: Path, label_scheme: str, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for mask_path in sorted(input_dir.glob("*.nii.gz")):
        output_path = output_dir / mask_path.name
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"{output_path} exists. Pass --overwrite to replace it.")
        mask, array = read_mask_array(mask_path)
        quadrant_array = remap_to_quadrants(array, label_scheme)
        write_like(mask, quadrant_array, output_path)
        print(f"[DONE] {mask_path.name} -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--label-scheme", choices=["sequential", "fdi"], default="sequential")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Import is intentionally exercised here so users get a clear missing
    # dependency error before any output files are touched.
    _ = sitk.Version()
    convert_directory(args.input_dir, args.output_dir, args.label_scheme, args.overwrite)


if __name__ == "__main__":
    main()
