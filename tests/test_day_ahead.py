"""Day-ahead roll-forward must match the direct forecast and must not peek ahead."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import rolling_origin_splits
from src.config import TempForecastConfig
from src.day_ahead import sarimax_day_ahead
from src.features import temperature_forecast_archive
from src.models import SarimaxForecaster


def _toy(n=260, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    temp = pd.Series(8 + 8 * np.cos(np.arange(n) / 40) + rng.normal(0, 2, n), index=idx)
    noise = np.cumsum(rng.normal(0, 5, n)) * 0.3
    y = pd.Series(1000 + 40 * np.maximum(15 - temp, 0) + noise
                  - 80 * (idx.dayofweek >= 5) + rng.normal(0, 10, n), index=idx)
    return y, temp


def _run(y, temp):
    archive = temperature_forecast_archive(temp, 3, TempForecastConfig(), seed=1)
    folds = list(rolling_origin_splits(y.index, y.index[200], step=20, horizon=3))
    fc = SarimaxForecaster(y, temp, archive, (1, 1, 1))
    first = [fc(f) for f in folds]
    return folds, first, sarimax_day_ahead(fc, folds)


def test_first_day_equals_direct_one_step_forecast():
    y, temp = _toy()
    folds, direct, da = _run(y, temp)
    for f, d in zip(folds, direct):
        row = da[da["date"] == f.test_index[0]].iloc[0]
        assert row["mean"] == pytest.approx(d["mean"].iloc[0], rel=1e-6)


def test_day_ahead_ignores_the_target_day_and_later():
    y, temp = _toy()
    folds, _, da = _run(y, temp)
    target = folds[0].origin + pd.Timedelta(days=5)
    y_mod = y.copy()
    y_mod.loc[target:] += 1000.0
    _, _, da_mod = _run(y_mod, temp)
    a = da.set_index("date").loc[target, "mean"]
    b = da_mod.set_index("date").loc[target, "mean"]
    assert a == pytest.approx(b, rel=1e-9)
