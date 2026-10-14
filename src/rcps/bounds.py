"""One-sided upper confidence bounds for the mean of bounded losses."""

from __future__ import annotations

import math
import numpy as np


def _losses(losses, delta: float) -> np.ndarray:
    x = np.asarray(losses, dtype=float).reshape(-1)
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("losses must be a non-empty finite vector")
    if np.any((x < 0) | (x > 1)):
        raise ValueError("losses must lie in [0, 1]")
    if not 0 < delta < 1:
        raise ValueError("delta must lie strictly between 0 and 1")
    return x


def hoeffding_ucb(losses, delta: float) -> float:
    """Hoeffding UCB: mean + sqrt(log(1/delta)/(2n)), clipped to [0,1]."""
    x = _losses(losses, delta)
    return float(min(1.0, x.mean() + math.sqrt(math.log(1 / delta) / (2 * x.size))))


def empirical_bernstein_ucb(losses, delta: float) -> float:
    """Empirical Bernstein UCB (Maurer-Pontil style), for bounded [0,1] data."""
    x = _losses(losses, delta)
    n = x.size
    if n < 2:
        return 1.0
    v = float(x.var(ddof=1))
    log = math.log(2 / delta)
    radius = math.sqrt(2 * v * log / n) + 7 * log / (3 * (n - 1))
    return float(min(1.0, x.mean() + radius))


def _wsr_log_capital(x: np.ndarray, risk: float, delta: float) -> float:
    """Log capital process used to invert the WSR risk bound."""
    n = x.size
    running_sum = 0.0
    squared_deviations = 0.0
    previous_variance = 0.25
    log_capital = 0.0
    log_delta = math.log(1 / delta)
    for i, loss in enumerate(x, start=1):
        running_sum += float(loss)
        mean_i = (0.5 + running_sum) / (i + 1)
        squared_deviations += (float(loss) - mean_i) ** 2
        variance_i = (0.25 + squared_deviations) / (i + 1)
        nu = min(1.0, math.sqrt(2 * log_delta / (n * previous_variance)))
        factor = 1.0 + nu * (risk - float(loss))
        if factor <= 0:
            return -math.inf
        log_capital += math.log(factor)
        previous_variance = variance_i
    return log_capital


def wsr_ucb(losses, delta: float, iterations: int = 64) -> float:
    """Invert the WSR confidence sequence to get an upper bound on mean risk.

    This is the bounded-loss WSR construction used in Bates et al. (2021).
    The returned value is in [0, 1].
    """
    x = _losses(losses, delta)
    lo, hi = 0.0, 1.0
    # The WSR capital is nondecreasing in the hypothesized risk R.
    for _ in range(iterations):
        mid = (lo + hi) / 2
        if _wsr_log_capital(x, mid, delta) > math.log(1 / delta):
            hi = mid
        else:
            lo = mid
    return float(hi)
