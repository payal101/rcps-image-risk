"""Image regression prediction intervals and per-image fractional miss loss."""

from __future__ import annotations
import numpy as np


def prediction_interval(prediction, radius: float, value_range=(0.0, 1.0)):
    """Return pixelwise symmetric intervals around a point prediction."""
    pred = np.asarray(prediction, dtype=float)
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    low, high = value_range
    return np.clip(pred - radius, low, high), np.clip(pred + radius, low, high)


def fractional_pixel_loss(targets, lower, upper) -> np.ndarray:
    """One scalar loss per image: fraction of pixels outside its intervals.

    Pixels within one image may be arbitrarily dependent. The statistical
    observations are the image pairs, and each image contributes one [0,1]
    loss to RCPS calibration.
    """
    y, lo, hi = np.asarray(targets), np.asarray(lower), np.asarray(upper)
    if y.shape != lo.shape or y.shape != hi.shape or y.ndim < 2:
        raise ValueError("targets/lower/upper must have matching batch-first shapes")
    missed = (y < lo) | (y > hi)
    return missed.reshape(missed.shape[0], -1).mean(axis=1)


def interval_loss_grid(targets, predictions, radii, value_range=(0.0, 1.0)) -> np.ndarray:
    """Build an (images, radii) matrix of fractional pixel miss losses."""
    radii = np.asarray(radii, dtype=float).reshape(-1)
    return np.column_stack([
        fractional_pixel_loss(targets, *prediction_interval(predictions, float(r), value_range))
        for r in radii
    ])
