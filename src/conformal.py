"""Calibrated prediction intervals from past out-of-sample errors (rolling split conformal).

At every test origin ``t`` the calibration set is made of backtest forecasts of the
same model whose *target date is <= t*, i.e. errors that were already observable when
the forecast was issued. Calibration data therefore start with the pre-test folds
(2019-09-29 .. 2020-12-13) and grow as the test period goes on (expanding window).

Two variants are implemented:

* ``"horizon"``: empirical quantiles of the signed errors ``e = y - mean`` of the same
  horizon ``h`` (about 22 values at the first test origin, 55 at the last one).
* ``"scaled"`` (primary, declared before looking at test results): normalised errors
  ``e / sd`` pooled over all horizons, where ``sd`` is the model's own predictive
  standard deviation. The model keeps the shape of its uncertainty across horizons
  and the conformal step rescales and re-centres it; pooling gives 14x more
  calibration points.

Both use the finite-sample split-conformal order statistic for each tail, so
under exchangeability each one-sided quantile is valid. Errors are not exchangeable
across a regime change, which is exactly what the regime tables examine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TAUS = np.round(np.arange(0.05, 0.951, 0.05), 2)  # 19 quantile levels for CRPS / pinball
MIN_CALIBRATION = 19  # smallest n for which the 5% and 95% order statistics exist


def qcol(tau: float) -> str:
    """Column name of the quantile forecast at level ``tau``."""
    return f"q{int(round(100 * tau)):02d}"


def implied_sd(lo: np.ndarray, hi: np.ndarray, level: float, floor: float = 1.0) -> np.ndarray:
    """Predictive standard deviation implied by a central normal interval [lo, hi]."""
    z = stats.norm.ppf(0.5 + level / 2)
    return np.maximum((np.asarray(hi) - np.asarray(lo)) / (2 * z), floor)


def conformal_quantile(scores: np.ndarray, tau: float) -> float:
    """Finite-sample split-conformal quantile of ``scores`` for one tail.

    Lower tail (tau < 0.5): the floor(tau (n+1))-th order statistic; upper tail:
    the ceil(tau (n+1))-th. For tau = 0.5 the sample median is used.
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    if n < MIN_CALIBRATION:
        raise ValueError(f"need at least {MIN_CALIBRATION} calibration scores, got {n}")
    if np.isclose(tau, 0.5):
        return float(np.median(s))
    if tau < 0.5:
        k = int(np.floor(tau * (n + 1)))
        return float(s[max(k, 1) - 1])
    k = int(np.ceil(tau * (n + 1)))
    return float(s[min(k, n) - 1])


def gaussian_quantiles(fc: pd.DataFrame, level: float, taus: np.ndarray = TAUS) -> pd.DataFrame:
    """Uncalibrated quantiles: the model's mean +- z_tau * implied sd (normal assumption)."""
    sd = implied_sd(fc["lo"], fc["hi"], level)
    out = fc.copy()
    for t in taus:
        out[qcol(t)] = fc["mean"].to_numpy() + stats.norm.ppf(t) * sd
    return out


def rolling_conformal(
    bt: pd.DataFrame,
    test_origins: pd.DatetimeIndex,
    level: float,
    mode: str = "scaled",
    taus: np.ndarray = TAUS,
) -> pd.DataFrame:
    """Recalibrate one model's backtest forecasts origin by origin.

    ``bt`` holds a single model's forecasts for calibration *and* test origins with
    columns origin, date, h, y, mean, lo, hi. Returns the rows of ``test_origins`` with
    calibrated ``lo``/``hi`` (the alpha/2 and 1-alpha/2 quantiles), the quantile grid
    ``q05 .. q95`` and the calibration size ``n_cal``.
    """
    if mode not in {"scaled", "horizon"}:
        raise ValueError("mode must be 'scaled' or 'horizon'")
    bt = bt.copy()
    bt["sd"] = implied_sd(bt["lo"], bt["hi"], level)
    bt["e"] = bt["y"] - bt["mean"]
    alpha = 1 - level
    parts = []
    for origin in test_origins:
        cur = bt[bt["origin"] == origin].copy()
        cal = bt[bt["date"] <= origin]  # errors observable at the origin
        if mode == "scaled":
            scores = (cal["e"] / cal["sd"]).to_numpy()
            qs = {t: conformal_quantile(scores, t) for t in taus}
            for t in taus:
                cur[qcol(t)] = cur["mean"] + qs[t] * cur["sd"]
            cur["n_cal"] = len(scores)
        else:
            groups = {h: g["e"].to_numpy() for h, g in cal.groupby("h")}
            for t in taus:
                cur[qcol(t)] = cur["mean"] + cur["h"].map(
                    lambda h, t=t: conformal_quantile(groups[h], t))
            cur["n_cal"] = cur["h"].map(lambda h: len(groups[h]))
        parts.append(cur)
    out = pd.concat(parts, ignore_index=True)
    out["lo"] = out[qcol(alpha / 2)]
    out["hi"] = out[qcol(1 - alpha / 2)]
    return out.drop(columns=["e"])


def online_conformal_daily(
    da: pd.DataFrame, test_dates: pd.DatetimeIndex, level: float, taus: np.ndarray = TAUS
) -> pd.DataFrame:
    """Scaled conformal for daily day-ahead forecasts: calibrate on all days before ``d``.

    The forecast for day d is issued on d-1 after observing y(d-1), so errors dated
    <= d-1 (strictly before d) are available.
    """
    da = da.sort_values("date").reset_index(drop=True)
    sd = implied_sd(da["lo"], da["hi"], level)
    scores = ((da["y"] - da["mean"]) / sd).to_numpy()
    dates = pd.DatetimeIndex(da["date"])
    alpha = 1 - level
    rows = []
    for d in test_dates:
        i = int(dates.get_loc(d))
        past = scores[dates < d]
        rec = {"date": d, "y": da.at[i, "y"], "mean": da.at[i, "mean"], "n_cal": len(past)}
        for t in taus:
            rec[qcol(t)] = da.at[i, "mean"] + conformal_quantile(past, t) * sd[i]
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["lo"] = out[qcol(alpha / 2)]
    out["hi"] = out[qcol(1 - alpha / 2)]
    return out
