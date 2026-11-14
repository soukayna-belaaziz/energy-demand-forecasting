"""End-to-end pipeline: data -> diagnostics -> SARIMAX spec validation -> rolling-origin
backtest -> combination, conformal intervals, regime analysis -> reports.

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
from src.combination import combine_backtest, combine_daily
from src.conformal import gaussian_quantiles, online_conformal_daily, rolling_conformal
from src.evaluation import dm_daily, dm_table, probabilistic_scores, regime_table
from src.metrics import coverage, mae, mape, rmse, score_table
from src.models import (
    LightGBMForecaster, make_ets, make_seasonal_naive, sarimax_exog,
)
from src.sarimax_selection import candidate_specs, select_sarimax

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
    """Overall / by-horizon / by-year point metrics on the test origins."""
    overall = score_table(bt, ["model"]).sort_values("MAE")
    _write(overall, "metrics_overall")
    by_h = score_table(bt, ["model", "h"])
    _write(by_h, "metrics_by_horizon")
    bt = bt.assign(year=bt["date"].dt.year)
    by_year = score_table(bt, ["model", "year"])[["MAE", "MAPE", "MASE", "coverage", "n"]]
    _write(by_year, "metrics_by_year")
    main = ["SARIMAX", "LightGBM", "ETS", "Seasonal naive", "Combination (inv_mse)"]
    plots.error_by_horizon(by_h.loc[[m for m in main if m in by_h.index.get_level_values(0)]],
                           level, config.FIG_DIR / "fig4_error_by_horizon.png")
    return {"overall": overall, "by_h": by_h, "by_year": by_year}


def interval_tables(
    bt_all: pd.DataFrame, test_origins: pd.DatetimeIndex, level: float, models: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Raw vs conformal intervals: overall scores, coverage by horizon, calibrated tables."""
    rows, by_h, calibrated = {}, [], {}
    for model in models:
        bm = bt_all[bt_all["model"] == model]
        raw = gaussian_quantiles(bm[bm["origin"].isin(test_origins)], level)
        variants = {
            "raw": raw,
            "conformal (horizon)": rolling_conformal(bm, test_origins, level, "horizon"),
            "conformal (scaled)": rolling_conformal(bm, test_origins, level, "scaled"),
        }
        calibrated[model] = variants["conformal (scaled)"].assign(model=model)
        for method, fc in variants.items():
            rows[(model, method)] = probabilistic_scores(fc, level)
            g = fc.groupby("h").apply(lambda d: pd.Series({
                "coverage": coverage(d["y"], d["lo"], d["hi"]),
                "width": float(np.mean(d["hi"] - d["lo"]))}), include_groups=False)
            by_h.append(g.assign(model=model, method=method).reset_index())
    overall = pd.DataFrame(rows).T
    overall.index.names = ["model", "method"]
    cov_h = pd.concat(by_h).set_index(["model", "method", "h"])
    return overall, cov_h, calibrated


def _rename_sarimax(name: str, chosen: str) -> str | None:
    if name == chosen:
        return "SARIMAX"
    if name == "baseline":
        return "SARIMAX baseline"
    return None


def main() -> None:
    t0 = time.time()
    np.random.seed(config.SEED)
    for p in (config.PROCESSED_DIR, config.FIG_DIR, config.TABLE_DIR):
        p.mkdir(parents=True, exist_ok=True)
    bt_cfg, tf_cfg = BacktestConfig(), TempForecastConfig()
    level = bt_cfg.interval_level

    df = build_dataset()
    df.to_csv(config.PROCESSED_DIR / "daily_gas_temperature.csv")
    _log(f"data: {len(df)} days {df.index[0].date()}..{df.index[-1].date()}", t0)

    order = diagnostics_step(df, bt_cfg, t0)

    archive = temperature_forecast_archive(df["temp"], bt_cfg.horizon, tf_cfg, config.SEED)
    all_folds = list(rolling_origin_splits(df.index, bt_cfg.calib_first_origin,
                                           bt_cfg.step_days, bt_cfg.horizon))
    first_test = pd.Timestamp(bt_cfg.first_origin)
    valid_folds = [f for f in all_folds if f.origin < first_test]
    folds = [f for f in all_folds if f.origin >= first_test]
    if valid_folds[-1].test_index[-1] > first_test or folds[0].origin != first_test:
        raise RuntimeError("validation folds must end before the first test origin")
    test_origins = pd.DatetimeIndex([f.origin for f in folds])
    y = df["total"]

    # 1. SARIMAX specification chosen on validation folds only
    specs = candidate_specs(order)
    chosen, sel_tbl, sar_fcs, bt_valid = select_sarimax(
        y, df["temp"], archive, specs, valid_folds, folds[0], level)
    _write(sel_tbl, "sarimax_spec_validation", ".4f")
    _log(f"SARIMAX spec validation done, chosen: {chosen.name}", t0)
    sarimax = sar_fcs[chosen.name]
    keep = {s: _rename_sarimax(s, chosen.name) for s in sar_fcs}
    keep = {k: v for k, v in keep.items() if v is not None}
    with warnings.catch_warnings():  # coefficient table of the chosen spec (with std. errors)
        warnings.simplefilter("ignore")
        f0 = folds[0]
        mod = sarimax._model(y.iloc[sarimax.burn_in : f0.train_end_pos + 1],
                             sarimax._x_real.iloc[sarimax.burn_in : f0.train_end_pos + 1])
        res0 = mod.fit(start_params=sarimax.results[f0.origin].params.to_numpy(), disp=False,
                       maxiter=300)
    (config.TABLE_DIR / "sarimax_chosen_fit.txt").write_text(str(res0.summary()))

    # 2. Backtest: test folds for all models, validation folds also for SARIMAX / LightGBM
    lgbm = LightGBMForecaster(y, df["temp"], archive, bt_cfg.horizon, level=level,
                              seed=config.SEED)
    parts = [bt_valid[bt_valid["model"].isin(keep)].replace({"model": keep})]
    for spec_name, label in keep.items():
        parts.append(run_backtest(y, folds, {label: sar_fcs[spec_name]}))
        _log(f"backtest {label}: {len(folds)} test origins", t0)
    parts.append(run_backtest(y, all_folds, {"LightGBM": lgbm}))
    _log(f"backtest LightGBM: {len(all_folds)} origins (validation + test)", t0)
    for name, fn in {"Seasonal naive": make_seasonal_naive(y, level=level),
                     "ETS": make_ets(y, level=level)}.items():
        parts.append(run_backtest(y, folds, {name: fn}))
        _log(f"backtest {name}: {len(folds)} test origins", t0)
    bt_all = pd.concat(parts, ignore_index=True)
    combos = []
    for scheme in ("equal", "inv_mse"):
        comb, w = combine_backtest(bt_all, ["SARIMAX", "LightGBM"], level, scheme)
        combos.append(comb)
        if scheme == "inv_mse":
            wt = w[w["origin"].isin(test_origins)].groupby("h")[["SARIMAX", "LightGBM"]].mean()
            _write(wt, "combination_weights_by_h")
    bt_all = pd.concat([bt_all, *combos], ignore_index=True)
    bt_all.to_csv(config.TABLE_DIR / "backtest_forecasts.csv", index=False)
    bt = bt_all[bt_all["origin"].isin(test_origins)].reset_index(drop=True)

    # 3. Point accuracy and Diebold-Mariano
    ev = evaluation_tables(bt, level)
    best_combo = min(["Combination (equal)", "Combination (inv_mse)"],
                     key=lambda m: ev["overall"].loc[m, "MAE"])
    dm = pd.concat([
        dm_table(bt, "SARIMAX", "LightGBM"),
        dm_table(bt, "SARIMAX", "SARIMAX baseline") if "SARIMAX baseline" in keep.values()
        else pd.DataFrame(),
        dm_table(bt, "Combination (equal)", "SARIMAX"),
        dm_table(bt, "Combination (inv_mse)", "SARIMAX"),
    ])
    _write(dm, "diebold_mariano", ".4f")

    # 4. Calibrated intervals (rolling split conformal), pinball loss and CRPS
    int_models = ["SARIMAX", "LightGBM", "Combination (equal)", "Combination (inv_mse)"]
    if "SARIMAX baseline" in keep.values():
        int_models.insert(1, "SARIMAX baseline")
    int_tbl, cov_h, calibrated = interval_tables(bt_all, test_origins, level, int_models)
    _write(int_tbl, "interval_calibration", ".3f")
    _write(cov_h.unstack("method"), "interval_coverage_by_horizon", ".3f")
    plots.calibration_by_horizon(cov_h, level, config.FIG_DIR / "fig7_interval_calibration.png")
    _log("conformal calibration done", t0)

    # 5. Regimes: 2021 pre-crisis vs 2022 crisis
    cal_all = pd.concat(calibrated.values(), ignore_index=True)
    reg = regime_table(bt, {"conformal": cal_all})
    _write(reg, "metrics_by_regime", ".3f")

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

    # 6. Operational day-ahead forecasts (h=1 for every day), calibration days included
    da_parts = [sarimax_day_ahead(sar_fcs[s], all_folds, level).assign(model=lab)
                for s, lab in keep.items()]
    da_parts.append(lightgbm_day_ahead(lgbm, all_folds).assign(model="LightGBM"))
    days_all = pd.DatetimeIndex(da_parts[0]["date"])
    da_parts.append(seasonal_naive_day_ahead(y, days_all).assign(model="Seasonal naive"))
    da = pd.concat(da_parts, ignore_index=True)
    da["y"] = y.loc[pd.DatetimeIndex(da["date"])].to_numpy()
    da["error"] = da["y"] - da["mean"]
    for scheme in ("equal", "inv_mse"):
        da = pd.concat([da, combine_daily(da, ["SARIMAX", "LightGBM"], scheme)],
                       ignore_index=True)
    test_days = days_all[days_all > first_test]
    da_conf = online_conformal_daily(da[da["model"] == "SARIMAX"], test_days, level)
    da = da[da["date"] > first_test].reset_index(drop=True)
    da.to_csv(config.TABLE_DIR / "day_ahead_forecasts.csv", index=False)
    da_tbl = da.groupby("model").apply(lambda g: pd.Series({
        "MAE": mae(g.y, g["mean"]), "RMSE": rmse(g.y, g["mean"]), "MAPE": mape(g.y, g["mean"]),
        "bias": float(g["error"].mean()), "n": len(g)}), include_groups=False).sort_values("MAE")
    for lab in keep.values():
        s_da = da[da.model == lab]
        da_tbl.loc[lab, "coverage_90_raw"] = coverage(s_da.y, s_da.lo, s_da.hi)
        da_tbl.loc[lab, "width_raw"] = float((s_da.hi - s_da.lo).mean())
    da_tbl.loc["SARIMAX", "coverage_90_conformal"] = coverage(da_conf.y, da_conf.lo, da_conf.hi)
    da_tbl.loc["SARIMAX", "width_conformal"] = float((da_conf.hi - da_conf.lo).mean())
    _write(da_tbl, "day_ahead_metrics")
    s_raw = gaussian_quantiles(da[da.model == "SARIMAX"], level)
    da_prob = pd.DataFrame({"raw": probabilistic_scores(s_raw, level),
                            "conformal (scaled)": probabilistic_scores(da_conf, level)}).T
    _write(da_prob, "day_ahead_interval_calibration", ".3f")
    da_dm = pd.concat([dm_daily(da, "SARIMAX", "LightGBM"),
                       dm_daily(da, "SARIMAX", "SARIMAX baseline")
                       if "SARIMAX baseline" in keep.values() else pd.DataFrame(),
                       dm_daily(da, "Combination (equal)", "SARIMAX"),
                       dm_daily(da, "Combination (inv_mse)", "SARIMAX")])
    _write(da_dm, "day_ahead_diebold_mariano", ".4f")
    da_reg = regime_table(da, {"conformal": da_conf.assign(model="SARIMAX")})
    _write(da_reg, "day_ahead_metrics_by_regime", ".3f")
    _log(f"day-ahead evaluation: {len(test_days)} test days ({len(days_all)} incl. validation)",
         t0)

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
    res = sarimax.results[last.origin]
    path = temps_known_at(df["temp"], archive, last.origin, bt_cfg.horizon)
    f = res.get_forecast(steps=bt_cfg.horizon, exog=sarimax.exog(path, last.test_index))
    bands = {}
    for lvl in (0.5, 0.8, 0.95):
        ci = f.conf_int(alpha=1 - lvl)
        bands[lvl] = pd.DataFrame({"lo": ci.iloc[:, 0].to_numpy(), "hi": ci.iloc[:, 1].to_numpy()},
                                  index=last.test_index)
    lgb_last = bt[(bt.model == "LightGBM") & (bt.origin == last.origin)].set_index("date")
    conf_last = calibrated["SARIMAX"]
    conf_last = conf_last[conf_last.origin == last.origin].set_index("date")[["lo", "hi"]]
    plots.fan_chart(hist, actual, bands, pd.Series(f.predicted_mean.to_numpy(), last.test_index),
                    lgb_last[["mean", "lo", "hi"]], config.FIG_DIR / "fig5_fan_chart.png",
                    conformal=conf_last)
    imp = pd.Series(lgbm.models["mean"].booster_.feature_importance("gain"), index=FEATURE_COLS)
    _write(imp.sort_values(ascending=False).to_frame("gain"), "lgbm_feature_importance", ".0f")
    plots.feature_importance(imp, config.FIG_DIR / "fig6_feature_importance.png")

    runtime = time.time() - t0
    summary = {
        "n_days": len(df), "start": str(df.index[0].date()), "end": str(df.index[-1].date()),
        "n_validation_origins": len(valid_folds),
        "first_validation_origin": str(valid_folds[0].origin.date()),
        "n_test_origins": len(folds), "first_test_origin": str(folds[0].origin.date()),
        "last_test_origin": str(folds[-1].origin.date()), "arima_order": order,
        "sarimax_spec": chosen.name, "best_combination": best_combo,
        "n_day_ahead_test_days": len(test_days), "runtime_seconds": round(runtime, 1),
    }
    (config.TABLE_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(sel_tbl.round(4).to_string())
    print(ev["overall"].round(3).to_string())
    print(dm.round(4).to_string())
    print(int_tbl.round(3).to_string())
    print(reg.round(3).to_string())
    print(da_tbl.round(3).to_string())
    print(da_prob.round(3).to_string())
    print(da_dm.round(4).to_string())
    print(cost_tbl.round(0).to_string())
    _log("pipeline finished", t0)


if __name__ == "__main__":
    main()
