"""Conformal intervals must use only errors observable at the origin and be calibrated."""
import numpy as np
import pandas as pd
import pytest

from src.conformal import (
    TAUS, conformal_quantile, implied_sd, online_conformal_daily, qcol, rolling_conformal,
)


def _toy_backtest(n_origins=60, horizon=5, sd_true=10.0, sd_model=4.0, seed=0):
    """Gaussian errors with sd_true while the model claims sd_model (too narrow)."""
    rng = np.random.default_rng(seed)
    rows = []
    origins = pd.date_range("2020-01-01", periods=n_origins, freq="7D")
    z = 1.6448536
    for o in origins:
        for h in range(1, horizon + 1):
            mean = 100.0
            y = mean + rng.normal(0, sd_true * np.sqrt(h))
            s = sd_model * np.sqrt(h)
            rows.append({"origin": o, "date": o + pd.Timedelta(days=h), "h": h, "y": y,
                         "mean": mean, "lo": mean - z * s, "hi": mean + z * s})
    return pd.DataFrame(rows)


def test_conformal_quantile_order_statistics():
    s = np.arange(1.0, 20.0)  # n = 19
    assert conformal_quantile(s, 0.05) == 1.0  # floor(0.05*20) = 1st
    assert conformal_quantile(s, 0.95) == 19.0  # ceil(0.95*20) = 19th
    with pytest.raises(ValueError):
        conformal_quantile(s[:10], 0.95)


def test_implied_sd_inverts_normal_interval():
    assert implied_sd(np.array([100 - 1.6448536 * 7]), np.array([100 + 1.6448536 * 7]),
                      0.9)[0] == pytest.approx(7.0, rel=1e-6)


@pytest.mark.parametrize("mode", ["scaled", "horizon"])
def test_conformal_repairs_undercoverage(mode):
    bt = _toy_backtest()
    test_origins = pd.DatetimeIndex(sorted(bt["origin"].unique())[25:])
    raw = bt[bt["origin"].isin(test_origins)]
    raw_cov = np.mean((raw.y >= raw.lo) & (raw.y <= raw.hi))
    out = rolling_conformal(bt, test_origins, 0.9, mode)
    cov = np.mean((out.y >= out.lo) & (out.y <= out.hi))
    assert raw_cov < 0.65
    assert 0.83 < cov < 0.97
    assert (out[[qcol(t) for t in TAUS]].diff(axis=1).iloc[:, 1:] >= 0).all().all()


@pytest.mark.parametrize("mode", ["scaled", "horizon"])
def test_conformal_ignores_errors_not_yet_observed(mode):
    bt = _toy_backtest()
    test_origins = pd.DatetimeIndex(sorted(bt["origin"].unique())[25:27])
    base = rolling_conformal(bt, test_origins, 0.9, mode)
    o = test_origins[0]
    corrupted = bt.copy()
    corrupted.loc[corrupted["date"] > o, "y"] += 1e4  # outcomes after the first origin
    out = rolling_conformal(corrupted, test_origins[:1], 0.9, mode)
    b = base[base.origin == o]
    np.testing.assert_allclose(out["lo"].to_numpy(), b["lo"].to_numpy())
    np.testing.assert_allclose(out["hi"].to_numpy(), b["hi"].to_numpy())


def test_online_daily_conformal_uses_only_past_days():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    da = pd.DataFrame({"date": dates, "mean": 0.0, "lo": -1.0, "hi": 1.0,
                       "y": rng.normal(0, 2, 200)})
    test = dates[150:]
    out = online_conformal_daily(da, test, 0.9)
    da2 = da.copy()
    da2.loc[da2["date"] >= test[0], "y"] = 1e6
    out2 = online_conformal_daily(da2, test[:1], 0.9)
    assert out2["hi"].iloc[0] == pytest.approx(out["hi"].iloc[0])
    assert out["n_cal"].iloc[0] == 150
