"""Combination weights: inverse-MSE, past errors only, equal-weight fallback."""
import numpy as np
import pandas as pd
import pytest

from src.combination import combine_backtest, inverse_mse_weights


def test_inverse_mse_weights():
    e = pd.DataFrame({"a": [1.0, -1.0] * 10, "b": [2.0, -2.0] * 10})
    w = inverse_mse_weights(e)
    assert w["a"] == pytest.approx(0.8)  # 1/1 vs 1/4 -> 0.8 / 0.2
    assert w.sum() == pytest.approx(1.0)
    assert inverse_mse_weights(e.iloc[:3])["a"] == pytest.approx(0.5)


def _bt(n_origins=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for o in pd.date_range("2020-01-01", periods=n_origins, freq="7D"):
        for h in (1, 2):
            y = 100 + rng.normal(0, 5)
            for model, sd in (("good", 1.0), ("bad", 10.0)):
                m = y + rng.normal(0, sd)
                rows.append({"model": model, "origin": o, "date": o + pd.Timedelta(days=h),
                             "h": h, "y": y, "mean": m, "lo": m - 5, "hi": m + 5,
                             "scale": 1.0, "error": y - m})
    return pd.DataFrame(rows)


def test_combination_equal_is_average_and_inv_mse_favours_better_model():
    bt = _bt()
    eq, _ = combine_backtest(bt, ["good", "bad"], 0.9, "equal")
    wide = bt.pivot_table(index=["origin", "date", "h"], columns="model", values="mean")
    np.testing.assert_allclose(eq["mean"].to_numpy(), wide.mean(axis=1).to_numpy())
    inv, w = combine_backtest(bt, ["good", "bad"], 0.9, "inv_mse", min_obs=10)
    late = w[w["origin"] == w["origin"].max()]
    assert (late["good"] > 0.9).all()
    early = w[w["origin"] == w["origin"].min()]
    assert np.allclose(early["good"], 0.5)  # no past errors yet -> equal weights


def test_inv_mse_weights_ignore_future_errors():
    bt = _bt()
    o = sorted(bt["origin"].unique())[15]
    _, w1 = combine_backtest(bt, ["good", "bad"], 0.9, "inv_mse")
    bt2 = bt.copy()
    mask = (bt2["date"] > o) & (bt2["model"] == "good")
    bt2.loc[mask, "error"] *= 100.0
    _, w2 = combine_backtest(bt2, ["good", "bad"], 0.9, "inv_mse")
    pd.testing.assert_frame_equal(w1[w1.origin == o].reset_index(drop=True),
                                  w2[w2.origin == o].reset_index(drop=True))
