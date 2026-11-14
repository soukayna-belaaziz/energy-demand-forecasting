"""Operational day-ahead (h=1) forecasts for every day of the test period.

The 14-day backtest only has one h=1 forecast every 21 days. A shipper, however,
nominates every day. Here each fold's fitted model (estimated on data up to its
origin) is rolled forward day by day until the next origin, using newly observed
consumption but *no re-estimation* of parameters. This yields one day-ahead
forecast per day without refitting the models 700 times.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import Fold
from src.features import build_direct_features, temps_known_at
from src.models import LightGBMForecaster, SarimaxForecaster


def _windows(folds: list[Fold], last_date: pd.Timestamp) -> list[tuple[Fold, pd.DatetimeIndex]]:
    """Days (origin_k, origin_{k+1}] are forecast with the model of fold k."""
    out = []
    for i, f in enumerate(folds):
        end = folds[i + 1].origin if i + 1 < len(folds) else last_date
        out.append((f, pd.date_range(f.origin + pd.Timedelta(days=1), end, freq="D")))
    return out


def sarimax_day_ahead(
    fc: SarimaxForecaster, folds: list[Fold], level: float = 0.9
) -> pd.DataFrame:
    """One-step-ahead SARIMAX forecasts using the forecast (not realised) temperature.

    The Kalman filter's one-step prediction is x_d'beta + E[u_d | data up to d-1].
    It is computed with realised exogenous values, then the regression part is
    swapped to the temperature forecast issued on d-1: + beta'(x_fc - x_realised).
    """
    y, temp, archive = fc.y, fc.temp, fc.archive
    x_real = fc.exog(temp, y.index)
    rows = []
    for fold, days in _windows(folds, y.index[-1]):
        res = fc.results[fold.origin]
        ext = res.extend(y.loc[days], exog=x_real.loc[days])
        pred = ext.get_prediction()
        ci = pred.conf_int(alpha=1 - level)
        beta = res.params[x_real.columns].to_numpy()
        x_fc = pd.concat(
            [fc.exog(temps_known_at(temp, archive, d - pd.Timedelta(days=1), 1),
                     pd.DatetimeIndex([d])) for d in days]
        )
        shift = (x_fc.to_numpy() - x_real.loc[days].to_numpy()) @ beta
        rows.append(pd.DataFrame({
            "date": days,
            "mean": pred.predicted_mean.to_numpy() + shift,
            "lo": ci.iloc[:, 0].to_numpy() + shift,
            "hi": ci.iloc[:, 1].to_numpy() + shift,
        }))
    return pd.concat(rows, ignore_index=True)


def lightgbm_day_ahead(fc: LightGBMForecaster, folds: list[Fold]) -> pd.DataFrame:
    """h=1 LightGBM forecasts from each fold's mean model (no quantile bands)."""
    rows = []
    for fold, days in _windows(folds, fc.y.index[-1]):
        model = fc.mean_models[fold.origin]
        origins = days - pd.Timedelta(days=1)
        x = build_direct_features(fc.y, fc.temp, fc.archive, origins, range(1, 2))
        mean = model.predict(x[fc.feature_cols]) + x["y_roll7"].to_numpy()
        rows.append(pd.DataFrame({"date": days, "mean": mean, "lo": np.nan, "hi": np.nan}))
    return pd.concat(rows, ignore_index=True)


def seasonal_naive_day_ahead(y: pd.Series, days: pd.DatetimeIndex, m: int = 7) -> pd.DataFrame:
    """y_hat(d) = y(d - m)."""
    mean = y.shift(m).loc[days].to_numpy()
    return pd.DataFrame({"date": days, "mean": mean, "lo": np.nan, "hi": np.nan})
