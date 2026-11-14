import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    coverage, diebold_mariano, interval_score, mae, mape, mase, rmse,
    score_table, seasonal_naive_scale,
)

Y = np.array([10.0, 20.0, 30.0, 40.0])
F = np.array([12.0, 18.0, 33.0, 40.0])


def test_point_metrics_known_values():
    assert mae(Y, F) == pytest.approx(7 / 4)
    assert rmse(Y, F) == pytest.approx(np.sqrt(17 / 4))
    assert mape(Y, F) == pytest.approx(100 * (0.2 + 0.1 + 0.1 + 0) / 4)


def test_mape_rejects_non_positive():
    with pytest.raises(ValueError):
        mape(np.array([0.0, 1.0]), np.array([1.0, 1.0]))


def test_mase_of_in_sample_seasonal_naive_is_one():
    rng = np.random.default_rng(0)
    y = rng.normal(100, 10, 70)
    scale = seasonal_naive_scale(y, m=7)
    assert mase(y[7:], y[:-7], scale) == pytest.approx(1.0)


def test_coverage_and_interval_score():
    lo, hi = Y - 1, Y + 1
    assert coverage(Y, lo, hi) == 1.0
    assert coverage(Y + 5, lo, hi) == 0.0
    assert interval_score(Y, lo, hi, 0.9) == pytest.approx(2.0)
    # a miss by 1 above the upper bound costs width + 2/alpha * 1
    assert interval_score(np.array([3.0]), np.array([0.0]), np.array([2.0]), 0.9) == \
        pytest.approx(2 + 20)


def test_diebold_mariano_sign_and_power():
    rng = np.random.default_rng(1)
    e_good = rng.normal(0, 1, 300)
    e_bad = rng.normal(0, 2, 300)
    stat, p = diebold_mariano(e_bad, e_good, h=1)
    assert stat > 0 and p < 0.01
    stat2, _ = diebold_mariano(e_good, e_bad, h=1)
    assert stat2 == pytest.approx(-stat)


def test_diebold_mariano_identical_errors():
    e = np.array([1.0, -2.0, 0.5, 3.0])
    assert diebold_mariano(e, e) == (0.0, 1.0)


def test_diebold_mariano_no_rejection_under_null():
    rng = np.random.default_rng(2)
    pvals = [diebold_mariano(rng.normal(size=200), rng.normal(size=200))[1] for _ in range(200)]
    assert 0.02 < np.mean(np.array(pvals) < 0.05) < 0.10


def test_score_table_groups():
    fc = pd.DataFrame({"model": ["a", "a", "b", "b"], "y": [10, 20, 10, 20],
                       "mean": [11, 19, 10, 20], "lo": [9, 18, 9, 19],
                       "hi": [12, 21, 11, 21], "scale": [1, 1, 1, 1]})
    t = score_table(fc, ["model"])
    assert t.loc["a", "MAE"] == 1 and t.loc["b", "MAE"] == 0
    assert t.loc["a", "coverage"] == 1.0


def test_pinball_and_crps_quantile_approximation():
    from scipy import stats

    from src.metrics import crps_from_quantiles, crps_gaussian, pinball_loss

    assert pinball_loss(np.array([10.0]), np.array([8.0]), 0.9) == pytest.approx(1.8)
    assert pinball_loss(np.array([10.0]), np.array([12.0]), 0.9) == pytest.approx(0.2)
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 4000)
    taus = np.linspace(0.01, 0.99, 99)
    q = np.tile(stats.norm.ppf(taus), (len(y), 1))
    exact = crps_gaussian(y, np.zeros_like(y), np.ones_like(y))
    assert crps_from_quantiles(y, q, taus) == pytest.approx(exact, rel=0.03)


def test_sarimax_spec_regressors():
    from src.models import SarimaxSpec, sarimax_exog

    idx = pd.date_range("2021-01-01", periods=20, freq="D")
    temp = pd.Series(np.linspace(-5, 15, 20), index=idx)
    base = sarimax_exog(temp, idx)
    nl = sarimax_exog(temp, idx, SarimaxSpec("nl", hdd_knots=(5.0,), hdd_weekend=True))
    assert set(nl.columns) - set(base.columns) == {"hdd_below5", "hdd_x_weekend"}
    weekday = idx.dayofweek < 5
    assert (nl.loc[weekday, "hdd_x_weekend"].dropna() == 0).all()
    assert (nl["hdd_below5"].dropna() >= 0).all()
