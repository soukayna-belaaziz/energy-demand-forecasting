"""The feature builder must never read information dated after the forecast origin."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import Fold
from src.config import TempForecastConfig
from src.features import (
    FEATURE_COLS,
    build_direct_features,
    temperature_forecast_archive,
    temps_known_at,
    weighted_temperature,
)
from src.models import LightGBMForecaster

H = 14


@pytest.fixture()
def toy():
    idx = pd.date_range("2020-01-01", periods=200, freq="D")
    rng = np.random.default_rng(0)
    y = pd.Series(1000 + 100 * np.sin(np.arange(200) / 10) + rng.normal(0, 5, 200), index=idx)
    temp = pd.Series(10 + 8 * np.cos(np.arange(200) / 30) + rng.normal(0, 1, 200), index=idx)
    archive = temperature_forecast_archive(temp, H, TempForecastConfig(), seed=1)
    return y, temp, archive


def _features(y, temp, archive, origin):
    x = build_direct_features(y, temp, archive, pd.DatetimeIndex([origin]), range(1, H + 1))
    return x[FEATURE_COLS]


def test_features_ignore_future_consumption(toy):
    y, temp, archive = toy
    origin = y.index[100]
    base = _features(y, temp, archive, origin)
    y_mod = y.copy()
    y_mod.loc[origin + pd.Timedelta(days=1):] *= 10  # corrupt everything after t
    pd.testing.assert_frame_equal(base, _features(y_mod, temp, archive, origin))


def test_features_ignore_future_realised_temperature(toy):
    y, temp, archive = toy
    origin = y.index[100]
    base = _features(y, temp, archive, origin)
    temp_mod = temp.copy()
    temp_mod.loc[origin + pd.Timedelta(days=1):] += 30  # future weather is unknown
    pd.testing.assert_frame_equal(base, _features(y, temp_mod, archive, origin))


def test_features_change_when_past_changes(toy):
    """Sanity check of the test above: past data must matter."""
    y, temp, archive = toy
    origin = y.index[100]
    y_mod = y.copy()
    y_mod.loc[origin] += 500
    assert not _features(y, temp, archive, origin).equals(_features(y_mod, temp, archive, origin))


def test_same_weekday_lag_is_not_in_the_future(toy):
    y, temp, archive = toy
    origin = y.index[100]
    x = build_direct_features(y, temp, archive, pd.DatetimeIndex([origin]), range(1, H + 1))
    for _, row in x.iterrows():
        back = 7 * int(np.ceil(row["h"] / 7))
        src_date = row["target_date"] - pd.Timedelta(days=back)
        assert src_date <= origin
        assert row["target_date"].dayofweek == src_date.dayofweek
        assert row["y_same_dow_last"] == pytest.approx(y.loc[src_date])


def test_training_rows_have_targets_before_origin(toy):
    y, temp, archive = toy
    origin_pos = 150
    fold = Fold(origin=y.index[origin_pos], train_end_pos=origin_pos,
                test_index=y.index[origin_pos + 1: origin_pos + 1 + H])
    tr = LightGBMForecaster(y, temp, archive, H).training_frame(fold)
    assert tr["target_date"].max() <= fold.origin
    assert tr["origin"].max() < fold.origin
    assert not tr["y_target"].isna().any()


def test_archive_noise_grows_with_lead(toy):
    _, temp, _ = toy
    long = pd.Series(np.zeros(5000), index=pd.date_range("2000-01-01", periods=5000))
    arc = temperature_forecast_archive(long, H, TempForecastConfig(sd_1=1.0, sd_slope=0.5), seed=3)
    sd = arc.iloc[:-H].std()
    assert sd[0] == 0  # issue-day temperature is observed, not forecast
    assert sd[1] == pytest.approx(1.0, rel=0.1)
    assert sd[H] == pytest.approx(1.0 + 0.5 * (H - 1), rel=0.1)


def test_temps_known_at_uses_forecasts_after_origin(toy):
    _, temp, archive = toy
    origin = temp.index[50]
    path = temps_known_at(temp, archive, origin, H)
    pd.testing.assert_series_equal(path.loc[:origin], temp.loc[:origin], check_names=False)
    assert np.allclose(path.iloc[-H:].to_numpy(), archive.loc[origin, 1:H].to_numpy())


def test_weighted_temperature_only_uses_past():
    t = pd.Series([0.0, 0.0, 0.0, 0.0, 15.0, 0.0])
    tw = weighted_temperature(t)
    assert tw.iloc[3] == 0.0  # the spike on day 4 does not affect day 3
    assert tw.iloc[4] == pytest.approx(15 / 1.875)
