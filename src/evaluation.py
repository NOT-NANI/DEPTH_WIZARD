"""Metrics and reports for paired height arrays."""
import numpy as np

from .calibration import ScaleShift, fit_scale_shift


def evaluate(prediction: np.ndarray, ground_truth: np.ndarray, valid: np.ndarray) -> tuple[ScaleShift, np.ndarray, dict]:
    if prediction.shape != ground_truth.shape or valid.shape != ground_truth.shape:
        raise ValueError("Prediction, ground truth, and validity mask must have matching shapes")
    mask = valid & np.isfinite(prediction) & np.isfinite(ground_truth)
    calibration = fit_scale_shift(prediction, ground_truth, mask)
    aligned = calibration.apply(prediction)
    errors = aligned[mask].astype(np.float64) - ground_truth[mask].astype(np.float64)
    corr = float(np.corrcoef(aligned[mask], ground_truth[mask])[0, 1])
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((ground_truth[mask] - ground_truth[mask].mean()) ** 2))
    metrics = {
        "method": "scale/shift aligned baseline",
        "valid_pixels": int(mask.sum()),
        "scale_coefficient": calibration.scale,
        "shift_coefficient_m": calibration.shift,
        "mae_m": float(np.mean(np.abs(errors))),
        "rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "pearson_r": corr,
        "r_squared": float(1 - ss_res / ss_tot) if ss_tot > 0 else None,
        "prediction_raw_min": float(np.min(prediction[mask])),
        "prediction_raw_max": float(np.max(prediction[mask])),
        "prediction_raw_mean": float(np.mean(prediction[mask])),
        "ground_truth_min_m": float(np.min(ground_truth[mask])),
        "ground_truth_max_m": float(np.max(ground_truth[mask])),
        "ground_truth_mean_m": float(np.mean(ground_truth[mask])),
        "ground_truth_median_m": float(np.median(ground_truth[mask])),
    }
    return calibration, aligned, metrics
