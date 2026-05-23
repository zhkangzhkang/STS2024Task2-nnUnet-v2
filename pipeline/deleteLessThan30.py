#!/usr/bin/env python3
"""Backward-compatible wrapper for 30 mm^3 post-processing."""

from __future__ import annotations

import argparse
from pathlib import Path

from postprocess_small_components import process_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mask_dir", type=Path, nargs="?", default=Path("outputs"))
    parser.add_argument("--threshold-mm3", type=float, default=30.0)
    args = parser.parse_args()
    process_dir(args.mask_dir, args.mask_dir, args.threshold_mm3, overwrite=False)


if __name__ == "__main__":
    main()
