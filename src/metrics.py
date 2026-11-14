"""Point-forecast metrics, interval coverage and the Diebold-Mariano test."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mae(y: np.ndarray, f: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(f))))


def rmse(y: np.ndarray, f: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(f)) ** 2)))


def mape(y: np.ndarray, f: np.ndarray) -> float:
    """Mean absolute percentage error in percent (y must be strictly positive)."""
    y = np.asarray(y, dtype=float)
    if np.any(y <= 0):
        raise ValueError("MAPE requires strictly positive actuals")
    return float(100 * np.mean(np.abs((y - np.asarray(f)) / y)))


def seasonal_naive_scale(y_train: np.ndarray, m: int = 7) -> float:
    """In-sample MAE of the seasonal naive forecast, the MASE denominator."""
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= m:
        raise ValueError("training series must be longer than the season length")
    return float(np.mean(np.abs(y_train[m:] - y_train[:-m])))


def mase(y: np.ndarray, f: np.ndarray, scale: float | np.ndarray) -> float:
    """Mean absolute scaled error; ``scale`` may vary per forecast (one per origin)."""
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(f)) / np.asarray(scale)))


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Share of actuals inside [lo, hi]."""
    y = np.asarray(y)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))


def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, level: float) -> float:
    """Gneiting-Raftery interval (Winkler) score, lower is better."""
    y, lo, hi = map(np.asarray, (y, lo, hi))
    alpha = 1 - level
    score = (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)
    return float(np.mean(score))


def diebold_mariano(
    e1: np.ndarray, e2: np.ndarray, h: int = 1, power: int = 2, max_lag: int | None = None
) -> tuple[float, float]:
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction.

    H0: equal expected loss. Loss = |e|^power. Positive statistic means model 1
    has the larger loss. The long-run variance uses autocovariances up to lag h-1.
    Returns (statistic, two-sided p-value from a Student-t with n-1 df).
    """
    d = np.abs(np.asarray(e1)) ** power - np.abs(np.asarray(e2)) ** power
    n = len(d)
    if n < 3:
        raise ValueError("need at least 3 loss differentials")
    d_bar = d.mean()
    if np.allclose(d, 0.0):
        return 0.0, 1.0
    dc = d - d_bar
    n_lags = h - 1 if max_lag is None else max_lag
    gamma = [np.sum(dc[k:] * dc[: n - k]) / n for k in range(n_lags + 1)]
    if max_lag is None:
        lrv = gamma[0] + 2 * sum(gamma[1:])
    else:  # Bartlett (Newey-West) weights keep the estimate positive
        lrv = gamma[0] + 2 * sum((1 - k / (n_lags + 1)) * g for k, g in enumerate(gamma[1:], 1))
    if lrv <= 0:
        lrv = gamma[0]
    dm = d_bar / np.sqrt(lrv / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = float(dm * hln)
    p = float(2 * stats.t.sf(abs(stat), df=n - 1))
    return stat, p


def score_table(fc: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Aggregate a long forecast table (y, mean, lo, hi, scale) into metrics."""
    def agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "MAE": mae(g["y"], g["mean"]),
                "RMSE": rmse(g["y"], g["mean"]),
                "MAPE": mape(g["y"], g["mean"]),
                "MASE": mase(g["y"], g["mean"], g["scale"]),
                "coverage": coverage(g["y"], g["lo"], g["hi"]),
                "width": float(np.mean(g["hi"] - g["lo"])),
                "n": len(g),
            }
        )
    return fc.groupby(by).apply(agg, include_groups=False)
