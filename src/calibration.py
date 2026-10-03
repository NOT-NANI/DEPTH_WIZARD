"""Scale and offset alignment; aligned outputs are sample-calibrated only."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScaleShift:
    scale: float
    shift: float

    def apply(self, prediction: np.ndarray) -> np.ndarray:
        return (self.scale * prediction + self.shift).astype(np.float32)


def fit_scale_shift(prediction: np.ndarray, ground_truth: np.ndarray, valid: np.ndarray) -> ScaleShift:
    mask = valid & np.isfinite(prediction) & np.isfinite(ground_truth)
    if int(mask.sum()) < 2:
        raise ValueError("At least two valid paired pixels are required")
    scale, shift = np.linalg.lstsq(np.column_stack((prediction[mask].astype(np.float64), np.ones(mask.sum()))), ground_truth[mask].astype(np.float64), rcond=None)[0]
    return ScaleShift(float(scale), float(shift))
