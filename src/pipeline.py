"""End-to-end pipeline: data -> diagnostics -> rolling-origin backtest -> reports.

Run with:  python -m src.pipeline
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src import config, plots
from src.backtest import rolling_origin_splits, run_backtest
from src.business import imbalance_cost
from src.day_ahead import lightgbm_day_ahead, sarimax_day_ahead, seasonal_naive_day_ahead
from src.config import BacktestConfig, TempForecastConfig
from src.data import build_dataset, load_imbalance_prices
from src.diagnostics import (
    ljung_box, mstl_decomposition, regression_residuals, select_arima_order,
    stationarity_tests,
)
from src.features import FEATURE_COLS, temperature_forecast_archive, temps_known_at
from src.metrics import diebold_mariano, mae, mape, rmse, score_table
from src.models import (
    LightGBMForecaster, SarimaxForecaster, make_ets, make_seasonal_naive, sarimax_exog,
)

BREAK_DATE = "2022-09-01"  # start of the visible post-crisis demand reduction
PORTFOLIO_SHARE = 0.01  # illustrative: a shipper with 1% of German demand


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def _write(df: pd.DataFrame, name: str, floatfmt: str = ".3f") -> None:
    df.to_csv(config.TABLE_DIR / f"{name}.csv")
    (config.TABLE_DIR / f"{name}.md").write_text(df.to_markdown(floatfmt=floatfmt) + "\n")


def diagnostics_step(df: pd.DataFrame, bt_cfg: BacktestConfig, t0: float) -> tuple[int, int, int]:
    """Decomposition, stationarity, order selection on the initial training window."""
    dec = mstl_decomposition(df["total"])
    plots.series_and_stl(df, dec, config.FIG_DIR / "fig1_series_mstl.png")
    plots.demand_vs_temperature(df, config.FIG_DIR / "fig2_demand_vs_temperature.png",
                                BREAK_DATE)

    train = df.loc[: bt_cfg.first_origin].iloc[3:]  # 3-day burn-in for weighted temp
    x = sarimax_exog(df["temp"], df.index).loc[train.index]
    resid = regression_residuals(train["total"], x)
    st = pd.DataFrame({
        "consumption (level)": stationarity_tests(train["total"]),
        "regression residual": stationarity_tests(resid),
        "diff. regression residual": stationarity_tests(resid.diff()),
    }).T
    _write(st, "stationarity_tests", ".4f")
    _log("stationarity tests done", t0)

    grid = select_arima_order(train["total"], x, d=1, max_p=2, max_q=2)
    _write(grid.set_index(["p", "d", "q"]), "arima_order_selection", ".3f")
    best = grid[grid["converged"]].iloc[0]
    order = (int(best.p), int(best.d), int(best.q))
    _log(f"ARIMA order grid done, BIC-best order {order}", t0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(train["total"], exog=x, order=order).fit(disp=False, maxiter=300)
    sar_resid = res.resid.iloc[8:]
    lb = ljung_box(sar_resid, [7, 14, 28], model_df=order[0] + order[2])
    _write(lb, "ljung_box_selected_model", ".4f")
    (config.TABLE_DIR / "sarimax_initial_fit.txt").write_text(str(res.summary()))
    plots.acf_pacf_panel(
        {
            "consumption (level)": train["total"],
            "first difference of regression residual": resid.diff(),
            f"SARIMAX{order} residual": sar_resid,
        },
        config.FIG_DIR / "fig3_acf_pacf.png",
    )
    return order


def evaluation_tables(bt: pd.DataFrame, level: float) -> dict:
    """Overall / by-horizon / by-year metrics and Diebold-Mariano tests."""
    overall = score_table(bt, ["model"]).sort_values("MAE")
    _write(overall, "metrics_overall")
    by_h = score_table(bt, ["model", "h"])
    _write(by_h, "metrics_by_horizon")
    bt = bt.assign(year=bt["date"].dt.year)
    by_year = score_table(bt, ["model", "year"])[["MAE", "MAPE", "MASE", "coverage", "n"]]
    _write(by_year, "metrics_by_year")
    plots.error_by_horizon(by_h, level, config.FIG_DIR / "fig4_error_by_horizon.png")

    best_two = list(overall.index[:2])
    a, b = best_two
    rows = []
    wide = bt.pivot_table(index=["origin", "h"], columns="model", values="error")
    for h in [1, 3, 7, 14]:
        e = wide.xs(h, level="h")
        for power, loss in [(2, "squared"), (1, "absolute")]:
            stat, p = diebold_mariano(e[a].to_numpy(), e[b].to_numpy(), h=1, power=power)
            rows.append({"horizon": h, "loss": loss, "DM_stat": stat, "p_value": p,
                         "n": len(e)})
    # pooled: mean loss over the 14-day window per origin; windows do not overlap
    for power, loss in [(2, "squared"), (1, "absolute")]:
        la = (wide[a].abs() ** power).groupby(level="origin").mean()
        lb_ = (wide[b].abs() ** power).groupby(level="origin").mean()
        d = (la - lb_).to_numpy()
        stat, p = _dm_on_losses(d)
        rows.append({"horizon": "1-14 pooled", "loss": loss, "DM_stat": stat, "p_value": p,
                     "n": len(d)})
    dm = pd.DataFrame(rows).set_index(["horizon", "loss"])
    _write(dm, "diebold_mariano", ".4f")
    return {"overall": overall, "by_h": by_h, "by_year": by_year, "dm": dm,
            "dm_models": best_two}


def _dm_on_losses(d: np.ndarray) -> tuple[float, float]:
    """DM test applied directly to a loss-differential series (lag-0 variance + HLN)."""
    from scipy import stats

    n = len(d)
    stat = float(d.mean() / np.sqrt(d.var() / n) * np.sqrt((n - 1) / n))
    return stat, float(2 * stats.t.sf(abs(stat), df=n - 1))


def main() -> None:
    t0 = time.time()
    np.random.seed(config.SEED)
    for p in (config.PROCESSED_DIR, config.FIG_DIR, config.TABLE_DIR):
        p.mkdir(parents=True, exist_ok=True)
    bt_cfg, tf_cfg = BacktestConfig(), TempForecastConfig()

    df = build_dataset()
    df.to_csv(config.PROCESSED_DIR / "daily_gas_temperature.csv")
    _log(f"data: {len(df)} days {df.index[0].date()}..{df.index[-1].date()}", t0)

    order = diagnostics_step(df, bt_cfg, t0)

    archive = temperature_forecast_archive(df["temp"], bt_cfg.horizon, tf_cfg, config.SEED)
    folds = list(rolling_origin_splits(df.index, bt_cfg.first_origin, bt_cfg.step_days,
                                       bt_cfg.horizon))
    y = df["total"]
    sarimax = SarimaxForecaster(y, df["temp"], archive, order, level=bt_cfg.interval_level)
    lgbm = LightGBMForecaster(y, df["temp"], archive, bt_cfg.horizon,
                              level=bt_cfg.interval_level, seed=config.SEED)
    forecasters = {
        "Seasonal naive": make_seasonal_naive(y, level=bt_cfg.interval_level),
        "ETS": make_ets(y, level=bt_cfg.interval_level),
        "SARIMAX": sarimax,
        "LightGBM": lgbm,
    }
    parts = []
    for name, fn in forecasters.items():
        parts.append(run_backtest(y, folds, {name: fn}))
        _log(f"backtest {name}: {len(folds)} origins", t0)
    bt = pd.concat(parts, ignore_index=True)
    bt.to_csv(config.TABLE_DIR / "backtest_forecasts.csv", index=False)

    ev = evaluation_tables(bt, bt_cfg.interval_level)

    # Sensitivity: SARIMAX with perfect (realised) future temperature
    s_bt = bt[bt["model"] == "SARIMAX"].copy()
    s_bt["oracle_mean"] = np.concatenate([sarimax.oracle_forecasts[o] for o in s_bt["origin"].unique()])
    sens = pd.DataFrame({
        "MAE": [mae(s_bt.y, s_bt["mean"]), mae(s_bt.y, s_bt.oracle_mean)],
        "RMSE": [rmse(s_bt.y, s_bt["mean"]), rmse(s_bt.y, s_bt.oracle_mean)],
        "MAPE": [mape(s_bt.y, s_bt["mean"]), mape(s_bt.y, s_bt.oracle_mean)],
    }, index=["simulated temperature forecast", "perfect temperature (oracle)"])
    _write(sens, "sarimax_temperature_sensitivity")
    sens_h = s_bt.groupby("h").apply(
        lambda g: pd.Series({"MAE_simulated": mae(g.y, g["mean"]),
                             "MAE_oracle": mae(g.y, g.oracle_mean)}), include_groups=False)
    _write(sens_h, "sarimax_temperature_sensitivity_by_h")

    # Operational day-ahead forecasts (h=1 for every test day)
    sar_da = sarimax_day_ahead(sarimax, folds, bt_cfg.interval_level)
    days = pd.DatetimeIndex(sar_da["date"])
    da = pd.concat([
        sar_da.assign(model="SARIMAX"),
        lightgbm_day_ahead(lgbm, folds).assign(model="LightGBM"),
        seasonal_naive_day_ahead(y, days).assign(model="Seasonal naive"),
    ], ignore_index=True)
    da["y"] = y.loc[pd.DatetimeIndex(da["date"])].to_numpy()
    da["error"] = da["y"] - da["mean"]
    da.to_csv(config.TABLE_DIR / "day_ahead_forecasts.csv", index=False)
    da_tbl = da.groupby("model").apply(lambda g: pd.Series({
        "MAE": mae(g.y, g["mean"]), "RMSE": rmse(g.y, g["mean"]), "MAPE": mape(g.y, g["mean"]),
        "bias": float(g["error"].mean()), "n": len(g)}), include_groups=False).sort_values("MAE")
    s_da = da[da.model == "SARIMAX"]
    da_tbl.loc["SARIMAX", "coverage_90"] = float(((s_da.y >= s_da.lo) & (s_da.y <= s_da.hi)).mean())
    _write(da_tbl, "day_ahead_metrics")
    wide_da = da.pivot(index="date", columns="model", values="error")
    dm_rows = []
    for power, loss in [(2, "squared"), (1, "absolute")]:
        stat, p = diebold_mariano(wide_da["SARIMAX"].to_numpy(), wide_da["LightGBM"].to_numpy(),
                                  h=1, power=power, max_lag=7)
        dm_rows.append({"comparison": "SARIMAX vs LightGBM", "loss": loss, "DM_stat": stat,
                        "p_value": p, "n": len(wide_da)})
    _write(pd.DataFrame(dm_rows).set_index(["comparison", "loss"]), "day_ahead_diebold_mariano",
           ".4f")
    _log(f"day-ahead evaluation: {len(days)} days", t0)

    # Business translation (illustrative, assumptions in src/business.py)
    prices = load_imbalance_prices()
    cost = imbalance_cost(da[["model", "date", "error"]], prices, PORTFOLIO_SHARE)
    cost_tbl = cost.groupby("model").agg(
        days=("cost_eur", "size"),
        mean_abs_error_MWh=("abs_error_mwh", "mean"),
        mean_cost_EUR_per_day=("cost_eur", "mean"),
        total_cost_EUR=("cost_eur", "sum"),
    ).sort_values("mean_cost_EUR_per_day")
    _write(cost_tbl, "imbalance_cost_illustrative", ".0f")
    spreads = pd.DataFrame({
        "pos_minus_index": prices["pos_price"] - prices["index_price"],
        "index_minus_neg": prices["index_price"] - prices["neg_price"],
    }).describe().T
    _write(spreads, "imbalance_spreads_EUR_MWh", ".2f")

    # Fan chart and feature importance for the last fold
    last = folds[-1]
    hist = y.iloc[last.train_end_pos - 27 : last.train_end_pos + 1]
    actual = y.loc[last.test_index]
    res = sarimax.last_result
    path = temps_known_at(df["temp"], archive, last.origin, bt_cfg.horizon)
    f = res.get_forecast(steps=bt_cfg.horizon, exog=sarimax_exog(path, last.test_index))
    bands = {}
    for lvl in (0.5, 0.8, 0.95):
        ci = f.conf_int(alpha=1 - lvl)
        bands[lvl] = pd.DataFrame({"lo": ci.iloc[:, 0].to_numpy(), "hi": ci.iloc[:, 1].to_numpy()},
                                  index=last.test_index)
    lgb_last = bt[(bt.model == "LightGBM") & (bt.origin == last.origin)].set_index("date")
    plots.fan_chart(hist, actual, bands, pd.Series(f.predicted_mean.to_numpy(), last.test_index),
                    lgb_last[["mean", "lo", "hi"]], config.FIG_DIR / "fig5_fan_chart.png")
    imp = pd.Series(lgbm.models["mean"].booster_.feature_importance("gain"), index=FEATURE_COLS)
    _write(imp.sort_values(ascending=False).to_frame("gain"), "lgbm_feature_importance", ".0f")
    plots.feature_importance(imp, config.FIG_DIR / "fig6_feature_importance.png")

    runtime = time.time() - t0
    summary = {
        "n_days": len(df), "start": str(df.index[0].date()), "end": str(df.index[-1].date()),
        "n_origins": len(folds), "first_origin": str(folds[0].origin.date()),
        "last_origin": str(folds[-1].origin.date()), "sarimax_order": order,
        "best_two_models": ev["dm_models"], "runtime_seconds": round(runtime, 1),
    }
    (config.TABLE_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(ev["overall"].round(3).to_string())
    print(ev["dm"].round(4).to_string())
    print(da_tbl.round(3).to_string())
    print(cost_tbl.round(0).to_string())
    _log("pipeline finished", t0)


if __name__ == "__main__":
    main()
