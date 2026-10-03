"""Held-out affine calibration and spatial transformation audit helpers."""
from __future__ import annotations

import numpy as np

from .calibration import ScaleShift, fit_scale_shift


def score_calibrated(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    valid: np.ndarray,
    calibration: ScaleShift,
) -> dict[str, float | int | None]:
    if prediction.shape != ground_truth.shape or valid.shape != ground_truth.shape:
        raise ValueError("Prediction, ground truth, and validity mask must share a shape")
    mask = valid & np.isfinite(prediction) & np.isfinite(ground_truth)
    if not mask.any():
        raise ValueError("Evaluation region contains no valid pixels")
    estimate = calibration.apply(prediction[mask]).astype(np.float64)
    target = ground_truth[mask].astype(np.float64)
    errors = estimate - target
    if np.std(estimate) == 0 or np.std(target) == 0:
        pearson = None
    else:
        pearson = float(np.corrcoef(estimate, target)[0, 1])
    residual = float(np.sum(errors**2))
    total = float(np.sum((target - target.mean())**2))
    return {
        "valid_pixels": int(mask.sum()),
        "mae_m": float(np.mean(np.abs(errors))),
        "rmse_m": float(np.sqrt(np.mean(errors**2))),
        "pearson_r": pearson,
        "r2": float(1 - residual / total) if total > 0 else None,
    }


def fit_on_calibration_region(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    valid: np.ndarray,
    calibration_mask: np.ndarray,
    evaluation_mask: np.ndarray,
) -> tuple[ScaleShift, dict[str, float | int | None], int]:
    if any(mask.shape != ground_truth.shape for mask in (valid, calibration_mask, evaluation_mask)):
        raise ValueError("All calibration and evaluation masks must match the height grid")
    calibration_valid = valid & calibration_mask & np.isfinite(prediction) & np.isfinite(ground_truth)
    evaluation_valid = valid & evaluation_mask & np.isfinite(prediction) & np.isfinite(ground_truth)
    if np.any(calibration_valid & evaluation_valid):
        raise ValueError("Calibration and evaluation pixels must be disjoint")
    fit = fit_scale_shift(prediction, ground_truth, calibration_valid)
    return fit, score_calibrated(prediction, ground_truth, evaluation_valid, fit), int(calibration_valid.sum())


def spatial_half_masks(shape: tuple[int, int], axis: str) -> tuple[np.ndarray, np.ndarray]:
    """Return two disjoint spatial halves: top/bottom (`row`) or left/right (`col`)."""
    height, width = shape
    if axis == "row":
        boundary = height // 2
        first = np.zeros(shape, dtype=bool); first[:boundary, :] = True
    elif axis == "col":
        boundary = width // 2
        first = np.zeros(shape, dtype=bool); first[:, :boundary] = True
    else:
        raise ValueError("axis must be 'row' or 'col'")
    return first, ~first


def transformed_predictions(prediction: np.ndarray) -> dict[str, np.ndarray]:
    """Return unmodified and flipped/rotated arrays without changing the source prediction."""
    return {
        "original": prediction,
        "horizontal_flip": np.fliplr(prediction),
        "vertical_flip": np.flipud(prediction),
        "rotate_180": np.rot90(prediction, 2),
    }
