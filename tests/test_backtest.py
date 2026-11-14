import numpy as np
import pandas as pd
import pytest

from src.backtest import rolling_origin_splits, run_backtest
from src.calendar_de import easter_sunday, german_national_holidays
from src.models import make_seasonal_naive

IDX = pd.date_range("2021-01-01", periods=100, freq="D")


def test_splits_are_expanding_and_non_overlapping():
    folds = list(rolling_origin_splits(IDX, "2021-02-01", step=10, horizon=14))
    assert len(folds) == 6  # origins at positions 31, 41, ..., 81 (85 is the last allowed)
    for f in folds:
        train = IDX[: f.train_end_pos + 1]
        assert train[-1] == f.origin
        assert f.test_index[0] == f.origin + pd.Timedelta(days=1)
        assert len(f.test_index) == 14
        assert train.intersection(f.test_index).empty
    ends = [f.train_end_pos for f in folds]
    assert all(b - a == 10 for a, b in zip(ends, ends[1:]))


def test_last_fold_has_full_horizon():
    folds = list(rolling_origin_splits(IDX, IDX[0], step=1, horizon=14))
    assert folds[-1].test_index[-1] == IDX[-1]


def test_splitter_rejects_impossible_origin():
    with pytest.raises(ValueError):
        list(rolling_origin_splits(IDX, IDX[-5], step=1, horizon=14))


def test_run_backtest_seasonal_naive_is_exact_on_periodic_series():
    y = pd.Series(np.tile(np.arange(7.0) + 1, 15)[:100], index=IDX)
    folds = list(rolling_origin_splits(IDX, IDX[30], step=7, horizon=14))
    out = run_backtest(y, folds, {"snaive": make_seasonal_naive(y)})
    assert np.allclose(out["error"], 0.0)
    assert set(out["h"]) == set(range(1, 15))


def test_easter_and_holidays():
    assert str(easter_sunday(2019)) == "2019-04-21"
    assert str(easter_sunday(2022)) == "2022-04-17"
    hol = german_national_holidays(2022)
    assert str(min(d for d, n in hol.items() if n == "Good Friday")) == "2022-04-15"
    assert len(german_national_holidays(2017)) == 10
