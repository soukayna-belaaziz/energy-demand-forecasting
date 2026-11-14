"""Forecasting models behind a common interface.

Every model is built as ``make_<model>(...) -> Forecaster`` where a forecaster maps
a backtest ``Fold`` to a DataFrame indexed by the test dates with columns
``mean``, ``lo``, ``hi``. Each forecaster only slices data up to ``fold.origin``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.backtest import Fold, Forecaster
from src.features import FEATURE_COLS, build_direct_features, exog_matrix, temps_known_at

MIN_HISTORY = 28  # rows needed before lag/rolling features are defined


def _z(level: float) -> float:
    return float(stats.norm.ppf(0.5 + level / 2))


# --------------------------------------------------------------------------- naive
def make_seasonal_naive(y: pd.Series, m: int = 7, level: float = 0.9) -> Forecaster:
    """y_hat(t+h) = y(t+h-m*k), k = ceil(h/m); normal intervals widening with k."""

    def fc(fold: Fold) -> pd.DataFrame:
        hist = y.iloc[: fold.train_end_pos + 1].to_numpy()
        h = np.arange(1, len(fold.test_index) + 1)
        k = np.ceil(h / m).astype(int)
        mean = hist[len(hist) - m * k + (h - 1) % m]
        resid = hist[m:] - hist[:-m]
        sigma = resid.std(ddof=1) * np.sqrt(k)
        z = _z(level)
        return pd.DataFrame(
            {"mean": mean, "lo": mean - z * sigma, "hi": mean + z * sigma},
            index=fold.test_index,
        )

    return fc


# --------------------------------------------------------------------------- ETS
def make_ets(y: pd.Series, m: int = 7, level: float = 0.9, window: int = 730) -> Forecaster:
    """Holt-Winters ETS(A,Ad,A) with weekly seasonality, fitted on the last ``window`` days.

    ETS has no temperature input: it is the "univariate" benchmark.
    """

    def fc(fold: Fold) -> pd.DataFrame:
        hist = y.iloc[max(0, fold.train_end_pos + 1 - window) : fold.train_end_pos + 1]
        hist = pd.Series(hist.to_numpy(), index=pd.RangeIndex(len(hist)))
        model = ETSModel(hist, error="add", trend="add", damped_trend=True,
                         seasonal="add", seasonal_periods=m)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit(disp=False, maxiter=500)
        n = len(fold.test_index)
        pred = res.get_prediction(start=len(hist), end=len(hist) + n - 1)
        sf = pred.summary_frame(alpha=1 - level)
        return pd.DataFrame(
            {"mean": sf["mean"].to_numpy(), "lo": sf["pi_lower"].to_numpy(),
             "hi": sf["pi_upper"].to_numpy()},
            index=fold.test_index,
        )

    return fc


# --------------------------------------------------------------------------- SARIMAX
def sarimax_exog(temp_path: pd.Series, index: pd.DatetimeIndex) -> pd.DataFrame:
    """HDD (weighted temperature), holiday/Christmas flags and weekday dummies."""
    x = exog_matrix(temp_path, index)
    for k in range(1, 7):
        x[f"dow{k}"] = (index.dayofweek == k).astype(float)
    return x


@dataclass
class SarimaxForecaster:
    """Regression with ARIMA errors; order fixed beforehand, parameters refitted per fold."""

    y: pd.Series
    temp: pd.Series
    archive: pd.DataFrame
    order: tuple[int, int, int]
    level: float = 0.9
    burn_in: int = 3  # weighted temperature needs 3 previous days
    _start_params: np.ndarray | None = field(default=None, repr=False)
    last_result: object | None = field(default=None, repr=False)
    oracle_forecasts: dict = field(default_factory=dict, repr=False)
    results: dict = field(default_factory=dict, repr=False)  # origin -> fitted results

    def __call__(self, fold: Fold) -> pd.DataFrame:
        y_tr = self.y.iloc[self.burn_in : fold.train_end_pos + 1]
        x_tr = sarimax_exog(self.temp, self.y.index)[self.burn_in : fold.train_end_pos + 1]
        n = len(fold.test_index)
        path = temps_known_at(self.temp, self.archive, fold.origin, n)
        x_te = sarimax_exog(path, fold.test_index)
        model = SARIMAX(y_tr, exog=x_tr, order=self.order)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # warm start from the previous fold's estimates speeds up re-estimation
            res = model.fit(start_params=self._start_params, disp=False, maxiter=300)
        self._start_params = res.params.to_numpy()
        self.last_result = res
        self.results[fold.origin] = res
        f = res.get_forecast(steps=n, exog=x_te)
        # Sensitivity only: same fitted model, but realised (perfect) future temperature.
        x_oracle = sarimax_exog(self.temp, self.y.index).loc[fold.test_index]
        self.oracle_forecasts[fold.origin] = res.get_forecast(
            steps=n, exog=x_oracle).predicted_mean.to_numpy()
        ci = f.conf_int(alpha=1 - self.level)
        return pd.DataFrame(
            {"mean": f.predicted_mean.to_numpy(), "lo": ci.iloc[:, 0].to_numpy(),
             "hi": ci.iloc[:, 1].to_numpy()},
            index=fold.test_index,
        )


# --------------------------------------------------------------------------- LightGBM
LGB_PARAMS: dict = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_bin": 63,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_jobs": 1,  # faster than 2 threads on a shared 2-CPU machine
    "verbose": -1,
}


@dataclass
class LightGBMForecaster:
    """Pooled direct multi-horizon LightGBM.

    One model for all horizons (``h`` is a feature). Target is expressed relative
    to the trailing 7-day mean at the origin (``y(t+h) - mean(y(t-6..t))``) so the
    trees do not have to extrapolate level shifts such as the 2022 demand drop.
    Prediction intervals come from two quantile-loss models.
    """

    y: pd.Series
    temp: pd.Series
    archive: pd.DataFrame
    horizon: int
    level: float = 0.9
    seed: int = 42
    feature_cols: list[str] = field(default_factory=lambda: list(FEATURE_COLS))
    quantiles: bool = True
    models: dict = field(default_factory=dict, repr=False)
    mean_models: dict = field(default_factory=dict, repr=False)  # origin -> fitted mean model

    def training_frame(self, fold: Fold) -> pd.DataFrame:
        """Rows whose origin AND target are both <= fold.origin (no look-ahead)."""
        origins = self.y.index[MIN_HISTORY : fold.train_end_pos]
        x = build_direct_features(self.y, self.temp, self.archive, origins,
                                  range(1, self.horizon + 1))
        return x[x["target_date"] <= fold.origin].reset_index(drop=True)

    def __call__(self, fold: Fold) -> pd.DataFrame:
        tr = self.training_frame(fold)
        y_rel = tr["y_target"] - tr["y_roll7"]
        alpha = (1 - self.level) / 2
        specs = {"mean": {"objective": "regression"}}
        if self.quantiles:
            specs["lo"] = {"objective": "quantile", "alpha": alpha}
            specs["hi"] = {"objective": "quantile", "alpha": 1 - alpha}
        te = build_direct_features(self.y, self.temp, self.archive,
                                   pd.DatetimeIndex([fold.origin]),
                                   range(1, len(fold.test_index) + 1))
        out = {}
        for name, spec in specs.items():
            model = lgb.LGBMRegressor(**LGB_PARAMS, **spec, random_state=self.seed)
            model.fit(tr[self.feature_cols], y_rel)
            self.models[name] = model
            out[name] = model.predict(te[self.feature_cols]) + te["y_roll7"].to_numpy()
        self.mean_models[fold.origin] = self.models["mean"]
        lo = np.minimum(out.get("lo", out["mean"]), out["mean"])
        hi = np.maximum(out.get("hi", out["mean"]), out["mean"])
        return pd.DataFrame({"mean": out["mean"], "lo": lo, "hi": hi}, index=fold.test_index)
