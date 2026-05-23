#!/usr/bin/env python3
"""Copy NIfTI volumes into nnU-Net v2 inference naming convention."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def strip_nii_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def normalize_case_id(path: Path) -> str:
    case_id = strip_nii_suffix(path)
    if case_id.endswith("_0000"):
        case_id = case_id[:-5]
    return case_id


def prepare(input_dir: Path, output_dir: Path, overwrite: bool) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in sorted(input_dir.glob("*.nii.gz")):
        case_id = normalize_case_id(src)
        dst = output_dir / f"{case_id}_0000.nii.gz"
        if dst.exists() and not overwrite:
            raise FileExistsError(f"{dst} exists. Pass --overwrite to replace it.")
        shutil.copy2(src, dst)
        count += 1
    if count == 0:
        raise RuntimeError(f"No .nii.gz files found in {input_dir}")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = prepare(args.input_dir, args.output_dir, args.overwrite)
    print(f"[DONE] Prepared {count} nnU-Net input images in {args.output_dir}")


if __name__ == "__main__":
    main()
