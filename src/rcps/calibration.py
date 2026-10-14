"""RCPS calibration for nested sets using a common calibration sample."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

from .bounds import empirical_bernstein_ucb, hoeffding_ucb, wsr_ucb

BOUND_METHODS: dict[str, Callable] = {
    "hoeffding": hoeffding_ucb,
    "empirical_bernstein": empirical_bernstein_ucb,
    "wsr": wsr_ucb,
}


@dataclass
class CalibrationResult:
    """Calibration summary; a missing selected index means no certificate."""
    parameters: np.ndarray
    empirical_risks: np.ndarray
    risk_ucbs: np.ndarray
    alpha: float
    delta: float
    method: str
    selected_index: int | None

    @property
    def certified(self) -> bool:
        return self.selected_index is not None

    @property
    def selected_parameter(self) -> float | None:
        return None if self.selected_index is None else float(self.parameters[self.selected_index])

    @property
    def selected_ucb(self) -> float | None:
        return None if self.selected_index is None else float(self.risk_ucbs[self.selected_index])


def calibrate_rcps(loss_matrix, parameters, alpha: float, delta: float,
                   method: str = "wsr") -> CalibrationResult:
    """Select the smallest nested-set parameter with a certified risk bound.

    ``loss_matrix[i, j]`` is the [0,1] loss for calibration example ``i`` at
    parameter ``j``. Parameters must be increasing set sizes and losses must
    be nonincreasing across each row. Pointwise bounds use the same calibration
    examples; suffix checking follows the monotone-family RCPS calibration
    rule and does not require a union bound over parameter values.
    """
    losses = np.asarray(loss_matrix, dtype=float)
    grid = np.asarray(parameters, dtype=float).reshape(-1)
    if losses.ndim != 2 or losses.shape[0] == 0 or losses.shape[1] != grid.size:
        raise ValueError("loss_matrix must have shape (n_examples, n_parameters)")
    if grid.size == 0 or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
        raise ValueError("parameters must be finite and strictly increasing")
    if not np.all(np.isfinite(losses)) or np.any((losses < 0) | (losses > 1)):
        raise ValueError("loss_matrix entries must be finite and in [0,1]")
    if np.any(np.diff(losses, axis=1) > 1e-10):
        raise ValueError("losses must be nonincreasing as nested sets grow")
    if not 0 < alpha < 1 or not 0 < delta < 1:
        raise ValueError("alpha and delta must lie strictly between 0 and 1")
    if method not in BOUND_METHODS:
        raise ValueError(f"method must be one of {tuple(BOUND_METHODS)}")

    bound_fn = BOUND_METHODS[method]
    empirical = losses.mean(axis=0)
    ucbs = np.array([bound_fn(losses[:, j], delta) for j in range(grid.size)])
    # Require every larger set to pass. This preserves monotone selection even
    # when noisy pointwise upper bounds are not themselves monotone.
    suffix_max = np.maximum.accumulate(ucbs[::-1])[::-1]
    passing = np.flatnonzero(suffix_max < alpha)
    selected = int(passing[0]) if passing.size else None
    return CalibrationResult(grid, empirical, ucbs, alpha, delta, method, selected)
