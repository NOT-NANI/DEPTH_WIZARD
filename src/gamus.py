"""Minimal, validated readers for the locally sampled GAMUS HDF5 files."""
from pathlib import Path

import h5py
import numpy as np

NODATA = -5.0


def load_image_h5(path: Path, expected_ndim: int | None = None) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        if "image" not in handle:
            raise KeyError(f"{path} has no 'image' dataset")
        array = handle["image"][:]
    if expected_ndim is not None and array.ndim != expected_ndim:
        raise ValueError(f"Expected {expected_ndim} dimensions, got {array.shape}")
    return array


def load_gamus(rgb_path: Path, height_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = load_image_h5(rgb_path, 3)
    height = load_image_h5(height_path, 2)
    if rgb.shape != (*height.shape, 3):
        raise ValueError(f"RGB {rgb.shape} and height {height.shape} shapes do not match")
    if rgb.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB data, got {rgb.dtype}")
    valid = np.isfinite(height) & (height != NODATA)
    if not valid.any():
        raise ValueError("No valid ground-truth heights found")
    return rgb, height.astype(np.float32, copy=False), valid


def print_statistics(rgb: np.ndarray, height: np.ndarray, valid: np.ndarray) -> None:
    values = height[valid]
    print(f"RGB shape/dtype: {rgb.shape} / {rgb.dtype}")
    print(f"Height shape/dtype: {height.shape} / {height.dtype}")
    print(f"NoData marker: {NODATA}; valid pixels: {int(valid.sum()):,}; NoData pixels: {int((~valid).sum()):,}")
    print(f"Valid AGL min/median/mean/max: {values.min():.6f} / {np.median(values):.6f} / {values.mean():.6f} / {values.max():.6f} m")
