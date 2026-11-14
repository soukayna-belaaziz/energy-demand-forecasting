"""Choose the LightGBM feature set (SARIMAX shown for reference) on validation folds that end before the test period.

Validation origins: 2019-10-06 .. 2020-12-13 (every 21 days, horizons 1-14).
The 2021-2022 test folds used in the main backtest are not touched here.

Run with:  python -m src.validate_lgbm
"""
from __future__ import annotations

import time

import pandas as pd

from src import config
from src.backtest import rolling_origin_splits, run_backtest
from src.config import BacktestConfig, TempForecastConfig
from src.data import build_dataset
from src.features import FEATURE_SETS, temperature_forecast_archive
from src.metrics import score_table
from src.models import LightGBMForecaster, SarimaxForecaster

VALID_FIRST_ORIGIN = "2019-10-06"
VALID_LAST_TARGET = "2021-01-03"  # = first test origin; validation targets stay before it


def main() -> None:
    t0 = time.time()
    cfg, tf = BacktestConfig(), TempForecastConfig()
    df = build_dataset().loc[:VALID_LAST_TARGET]
    archive = temperature_forecast_archive(build_dataset()["temp"], cfg.horizon, tf,
                                           config.SEED).loc[df.index]
    folds = list(rolling_origin_splits(df.index, VALID_FIRST_ORIGIN, cfg.step_days, cfg.horizon))
    parts = []
    for name, cols in FEATURE_SETS.items():
        f = LightGBMForecaster(df["total"], df["temp"], archive, cfg.horizon,
                               seed=config.SEED, feature_cols=cols, quantiles=False)
        parts.append(run_backtest(df["total"], folds, {name: f}))
    # reference: the SARIMAX specification used in the main backtest (order (1,1,2))
    sar = SarimaxForecaster(df["total"], df["temp"], archive, (1, 1, 2))
    parts.append(run_backtest(df["total"], folds, {"SARIMAX (reference)": sar}))
    bt = pd.concat(parts, ignore_index=True)
    overall = score_table(bt, ["model"])[["MAE", "RMSE", "MAPE", "MASE", "n"]]
    by_h = score_table(bt, ["model", "h"])["MAE"].unstack(0).loc[[1, 2, 3, 7, 14]]
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    text = (f"Validation folds: {len(folds)} origins {folds[0].origin.date()} .. "
            f"{folds[-1].origin.date()}\n\n" + overall.to_markdown(floatfmt=".3f")
            + "\n\nMAE by horizon\n\n" + by_h.to_markdown(floatfmt=".1f") + "\n")
    (config.TABLE_DIR / "lgbm_feature_set_validation.md").write_text(text)
    print(text)
    print(f"runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
