"""Leakage-safe feature construction.

Two ideas keep the features honest:

1. *Simulated temperature-forecast archive.* At forecast origin ``t`` the
   temperature of day ``t+h`` is unknown. We therefore never feed the realised
   temperature of a future day to a model. Instead ``temperature_forecast_archive``
   creates, for every issue date ``t`` and lead ``h``, a forecast equal to the
   realised value plus noise whose spread grows with ``h`` (an explicit assumption,
   see ``config.TempForecastConfig``).
2. *Origin-anchored lags.* For a forecast issued at ``t`` for day ``t+h``, all
   consumption features are taken at dates ``<= t`` (direct multi-horizon strategy).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.calendar_de import calendar_frame
from src.config import HDD_BASE, TempForecastConfig

WEIGHTS = np.array([1.0, 0.5, 0.25, 0.125])  # BDEW-style weighted temperature
N_LAGS = 14


def temperature_forecast_archive(
    temp: pd.Series, horizon: int, cfg: TempForecastConfig, seed: int
) -> pd.DataFrame:
    """Simulated archive: row = issue date t, column h = forecast for t+h (h=1..horizon).

    Column 0 holds the realised temperature of the issue day itself (known at t).
    Forecasts beyond the end of the temperature series are NaN.
    """
    rng = np.random.default_rng(seed)
    n = len(temp)
    values = temp.to_numpy()
    out = np.full((n, horizon + 1), np.nan)
    out[:, 0] = values
    for h in range(1, horizon + 1):
        sd = cfg.sd_1 + cfg.sd_slope * (h - 1)
        noise = rng.normal(0.0, sd, size=n)
        shifted = np.full(n, np.nan)
        shifted[: n - h] = values[h:]
        out[:, h] = shifted + noise
    return pd.DataFrame(out, index=temp.index, columns=range(horizon + 1))


def temps_known_at(
    temp: pd.Series, archive: pd.DataFrame, origin: pd.Timestamp, horizon: int
) -> pd.Series:
    """Temperature path as known at ``origin``: realised up to t, forecast for t+1..t+H."""
    past = temp.loc[:origin]
    fut_idx = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon, freq="D")
    fut = pd.Series(archive.loc[origin, 1 : horizon].to_numpy(), index=fut_idx)
    return pd.concat([past, fut])


def weighted_temperature(temp: pd.Series) -> pd.Series:
    """Geometric weighting of today and the 3 previous days (thermal inertia of buildings)."""
    parts = [temp.shift(j) * w for j, w in enumerate(WEIGHTS)]
    return (sum(parts) / WEIGHTS.sum()).rename("temp_w")


def exog_matrix(temp_path: pd.Series, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Exogenous regressors for SARIMAX on ``index`` from a temperature path."""
    tw = weighted_temperature(temp_path).reindex(index)
    cal = calendar_frame(index)
    return pd.DataFrame(
        {
            "hdd_w": np.maximum(HDD_BASE - tw, 0.0),
            "is_holiday": cal["is_holiday"],
            "xmas_period": cal["xmas_period"],
        },
        index=index,
    )


def _target_temperature_block(
    temp: pd.Series, archive: pd.DataFrame, origins: np.ndarray, h: int
) -> dict[str, np.ndarray]:
    """Temperature features for target day t+h using only information available at t."""
    t_vals = temp.to_numpy()
    arc = archive.to_numpy()
    w_parts = []
    for j, w in enumerate(WEIGHTS):
        lead = h - j  # day t+h-j relative to origin t
        if lead >= 1:
            v = arc[origins, lead]
        else:
            idx = origins + lead
            v = np.where(idx >= 0, t_vals[np.clip(idx, 0, None)], np.nan)
        w_parts.append(w * v)
    tw = np.sum(w_parts, axis=0) / WEIGHTS.sum()
    return {
        "temp_fc_target": arc[origins, h],
        "temp_w_target": tw,
        "hdd_w_target": np.maximum(HDD_BASE - tw, 0.0),
        "temp_origin": t_vals[origins],
    }


def build_direct_features(
    y: pd.Series,
    temp: pd.Series,
    archive: pd.DataFrame,
    origins: pd.DatetimeIndex,
    horizons: range,
) -> pd.DataFrame:
    """Rows = (origin t, horizon h); features use only data dated <= t (plus calendar of t+h).

    ``y`` may contain values after the origins; they are never read for features
    (the target column ``y_target`` is the only exception and is returned separately
    so training code can drop rows whose target lies beyond the training cut-off).
    """
    pos = y.index.get_indexer(origins)
    if (pos < 0).any():
        raise KeyError("origins must be dates of y")
    yv = y.to_numpy()
    n = len(yv)
    lag_mat = np.full((len(pos), N_LAGS), np.nan)
    for k in range(N_LAGS):
        ok = pos - k >= 0
        lag_mat[ok, k] = yv[pos[ok] - k]
    roll7 = pd.Series(yv).rolling(7).mean().to_numpy()[pos]
    roll28 = pd.Series(yv).rolling(28).mean().to_numpy()[pos]

    frames = []
    for h in horizons:
        tgt_pos = pos + h
        tgt_dates = origins + pd.Timedelta(days=h)
        cal = calendar_frame(pd.DatetimeIndex(tgt_dates))
        # most recent observed value on the same weekday as the target day
        back = 7 * int(np.ceil(h / 7))
        same_dow_pos = tgt_pos - back
        same_dow = np.where(same_dow_pos >= 0, yv[np.clip(same_dow_pos, 0, None)], np.nan)
        block = {
            "origin": origins,
            "target_date": tgt_dates,
            "h": h,
            **{f"y_lag{k}": lag_mat[:, k] for k in range(N_LAGS)},
            "y_roll7": roll7,
            "y_roll28": roll28,
            "y_same_dow_last": same_dow,
            **_target_temperature_block(temp, archive, pos, h),
            **{c: cal[c].to_numpy() for c in cal.columns},
            "y_target": np.where(tgt_pos < n, yv[np.clip(tgt_pos, None, n - 1)], np.nan),
        }
        frames.append(pd.DataFrame(block))
    return pd.concat(frames, ignore_index=True)


FEATURE_COLS: list[str] = (
    ["h"]
    + [f"y_lag{k}" for k in range(N_LAGS)]
    + ["y_roll7", "y_roll28", "y_same_dow_last",
       "temp_fc_target", "temp_w_target", "hdd_w_target", "temp_origin",
       "dow", "is_weekend", "is_holiday", "xmas_period", "doy_sin", "doy_cos"]
)
