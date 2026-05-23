#!/usr/bin/env python3
"""Merge quadrant tooth predictions back into full-volume masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk


FDI_STS_MAPPING = {
    1: list(range(21, 29)),
    2: list(range(11, 19)),
    3: list(range(41, 49)),
    4: list(range(31, 39)),
}

SEQUENTIAL_MAPPING = {
    1: list(range(1, 9)),
    2: list(range(9, 17)),
    3: list(range(17, 25)),
    4: list(range(25, 33)),
}


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
    raise FileNotFoundError(f"No image found for case {case_id} in {image_dir}")


def adjust_size(source_array: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    result = np.zeros(target_shape, dtype=source_array.dtype)
    min_shape = [min(source_dim, target_dim) for source_dim, target_dim in zip(source_array.shape, target_shape)]
    src_slices = tuple(slice(0, dim) for dim in min_shape)
    dst_slices = tuple(slice(0, dim) for dim in min_shape)
    result[dst_slices] = source_array[src_slices]
    return result


def label_mapping(output_label_scheme: str) -> dict[int, list[int]]:
    if output_label_scheme == "fdi":
        return FDI_STS_MAPPING
    if output_label_scheme == "sequential":
        return SEQUENTIAL_MAPPING
    raise ValueError(f"Unsupported output label scheme: {output_label_scheme}")


def remap_local_quadrant_labels(
    local_mask: np.ndarray,
    quadrant: int,
    output_label_scheme: str,
) -> np.ndarray:
    mapping = label_mapping(output_label_scheme)
    remapped = np.zeros(local_mask.shape, dtype=np.uint16)
    for local_label in np.unique(local_mask):
        local_label = int(local_label)
        if local_label in (0, 9, 15):
            continue
        if local_label < 1 or local_label > 8:
            print(f"[WARN] unexpected local label {local_label} in quadrant {quadrant}; ignored")
            continue
        remapped[local_mask == local_label] = mapping[quadrant][local_label - 1]
    return remapped


def merge_quadrants(
    image_dir: Path,
    quadrant_mask_dir: Path,
    resizer_dir: Path,
    output_dir: Path,
    output_label_scheme: str = "fdi",
    mask_suffix: str = "_Mask",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for resizer_path in sorted(resizer_dir.glob("*.npy")):
        case_id = resizer_path.stem
        image = sitk.ReadImage(str(find_image(image_dir, case_id)))
        image_array = sitk.GetArrayFromImage(image)
        merged = np.zeros(image_array.shape, dtype=np.uint16)

        resizer_dict = dict(np.load(resizer_path, allow_pickle=True).tolist())
        for quadrant in range(1, 5):
            crop_input_name = f"{case_id}_quadrant_{quadrant}_0000.nii.gz"
            crop_prediction = quadrant_mask_dir / f"{case_id}_quadrant_{quadrant}.nii.gz"
            if crop_input_name not in resizer_dict:
                print(f"[WARN] {case_id}: no crop metadata for quadrant {quadrant}; skipped")
                continue
            if not crop_prediction.exists():
                print(f"[WARN] {case_id}: missing prediction {crop_prediction.name}; skipped")
                continue

            local_image = sitk.ReadImage(str(crop_prediction))
            local_mask = sitk.GetArrayFromImage(local_image)
            if not np.issubdtype(local_mask.dtype, np.integer):
                local_mask = np.rint(local_mask).astype(np.int16)

            remapped = remap_local_quadrant_labels(local_mask, quadrant, output_label_scheme)
            slices = resizer_dict[crop_input_name]
            remapped = adjust_size(remapped, merged[slices].shape)
            overwrite = remapped > 0
            merged_region = merged[slices]
            merged_region[overwrite] = remapped[overwrite]
            merged[slices] = merged_region

        out = sitk.GetImageFromArray(merged)
        out.CopyInformation(image)
        out_path = output_dir / f"{case_id}{mask_suffix}.nii.gz"
        sitk.WriteImage(out, str(out_path))
        print(f"[DONE] wrote {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path, help="Original nnU-Net-style input image directory.")
    parser.add_argument("quadrant_mask_dir", type=Path, help="Predicted tooth masks for quadrant crops.")
    parser.add_argument("resizer_dir", type=Path, help="Crop metadata produced by quadrant_locate.py.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--output-label-scheme", choices=["fdi", "sequential"], default="fdi")
    parser.add_argument("--mask-suffix", default="_Mask")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_quadrants(
        args.image_dir,
        args.quadrant_mask_dir,
        args.resizer_dir,
        args.output_dir,
        output_label_scheme=args.output_label_scheme,
        mask_suffix=args.mask_suffix,
    )


if __name__ == "__main__":
    main()
