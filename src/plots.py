"""Static figures (matplotlib), one function per figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # noqa: E402

# Categorical palette (fixed order, one hue per model; validated reference palette)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODEL_ORDER = ["SARIMAX", "LightGBM", "ETS", "Seasonal naive", "Combination (inv_mse)"]
MODEL_COLOR = dict(zip(MODEL_ORDER, PALETTE))
MODEL_COLOR["Combination (equal)"] = MODEL_COLOR["Combination (inv_mse)"]
MODEL_COLOR["SARIMAX baseline"] = INK_2

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 130, "font.size": 9.5,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK_2, "ytick.color": INK_2, "legend.frameon": False,
    "lines.linewidth": 1.4, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def series_and_stl(df: pd.DataFrame, dec: pd.DataFrame, path: Path) -> None:
    """Consumption, temperature and MSTL components (separate panels, no dual axis)."""
    fig, axes = plt.subplots(5, 1, figsize=(9, 9.5), sharex=True)
    axes[0].plot(df.index, df["total"], color=PALETTE[0], lw=0.8)
    axes[0].set_title("German daily natural-gas consumption (GWh/day)")
    axes[1].plot(df.index, df["temp"], color=PALETTE[1], lw=0.8)
    axes[1].set_title("Germany country-average 2 m temperature, ERA5 (deg C)")
    axes[2].plot(dec.index, dec["trend"], color=INK, lw=1.4)
    axes[2].set_title("MSTL trend")
    axes[3].plot(dec.index, dec["seasonal_365"], color=PALETTE[2], lw=1.0)
    axes[3].set_title("MSTL annual seasonal component")
    axes[4].plot(dec.index, dec["seasonal_7"], color=PALETTE[0], lw=0.5)
    axes[4].plot(dec.index, dec["resid"], color=INK_2, lw=0.5, alpha=0.6)
    axes[4].set_title("MSTL weekly component (blue) and remainder (grey)")
    for ax in axes[2:]:
        ax.set_ylabel("GWh/day")
    _save(fig, path)


def demand_vs_temperature(df: pd.DataFrame, path: Path, break_date: str) -> None:
    """Scatter of consumption against temperature, before/after the 2022 demand drop."""
    fig, ax = plt.subplots(figsize=(7, 4.3))
    pre, post = df.loc[: break_date], df.loc[break_date:]
    ax.scatter(pre["temp"], pre["total"], s=8, color=PALETTE[0], alpha=0.45,
               edgecolor="none", label=f"until {break_date}")
    ax.scatter(post["temp"], post["total"], s=10, color=PALETTE[1], alpha=0.8,
               edgecolor="none", label=f"from {break_date}")
    ax.set_xlabel("Daily mean temperature (deg C)")
    ax.set_ylabel("Consumption (GWh/day)")
    ax.set_title("Gas demand is driven by heating: near-linear below ~15 deg C")
    ax.legend(loc="upper right")
    _save(fig, path)


def acf_pacf_panel(series: dict[str, pd.Series], path: Path, lags: int = 28) -> None:
    """ACF (left) and PACF (right) for several series (rows)."""
    fig, axes = plt.subplots(len(series), 2, figsize=(9, 2.4 * len(series)))
    axes = np.atleast_2d(axes)
    for i, (name, s) in enumerate(series.items()):
        s = s.dropna()
        plot_acf(s, lags=lags, ax=axes[i, 0], color=PALETTE[0], vlines_kwargs={"colors": PALETTE[0]})
        plot_pacf(s, lags=lags, ax=axes[i, 1], method="ywm", color=PALETTE[0],
                  vlines_kwargs={"colors": PALETTE[0]})
        axes[i, 0].set_title(f"ACF: {name}")
        axes[i, 1].set_title(f"PACF: {name}")
        for ax in axes[i]:
            ax.set_ylim(-1.05, 1.05)
            ax.set_xlabel("lag (days)")
    _save(fig, path)


def error_by_horizon(by_h: pd.DataFrame, level: float, path: Path) -> None:
    """MAE and interval coverage by forecast horizon, one line per model."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for model in MODEL_ORDER:
        if model not in by_h.index.get_level_values(0):
            continue
        g = by_h.loc[model]
        c = MODEL_COLOR[model]
        axes[0].plot(g.index, g["MAE"], color=c, lw=2, marker="o", ms=3.5, label=model)
        axes[0].annotate(model, (g.index[-1], g["MAE"].iloc[-1]), xytext=(5, 0),
                         textcoords="offset points", va="center", color=INK_2, fontsize=8.5)
        axes[1].plot(g.index, 100 * g["coverage"], color=c, lw=2, marker="o", ms=3.5,
                     label=model)
    axes[0].set_title("Backtest MAE by horizon")
    axes[0].set_xlabel("horizon (days ahead)")
    axes[0].set_ylabel("MAE (GWh/day)")
    axes[0].set_xlim(0.5, 17.5)
    axes[0].set_ylim(bottom=0)
    axes[1].axhline(100 * level, color=INK, lw=1, ls="--")
    axes[1].annotate(f"nominal {level:.0%}", (14, 100 * level), xytext=(0, 4),
                     textcoords="offset points", ha="right", color=INK_2, fontsize=8.5)
    axes[1].set_title(f"Empirical coverage of {level:.0%} prediction intervals")
    axes[1].set_xlabel("horizon (days ahead)")
    axes[1].set_ylabel("coverage (%)")
    axes[1].legend(loc="lower left", ncol=2)
    _save(fig, path)


def fan_chart(
    history: pd.Series, actual: pd.Series, sarimax_bands: dict[float, pd.DataFrame],
    sarimax_mean: pd.Series, lgbm: pd.DataFrame, path: Path,
    conformal: pd.DataFrame | None = None,
) -> None:
    """Fan chart for one origin: SARIMAX nested intervals (left), LightGBM quantiles (right).

    ``conformal`` (lo/hi) is drawn as the dashed outline of the calibrated 90% interval.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for ax in axes:
        ax.plot(history.index, history, color=INK, lw=1.4, label="observed (training)")
        ax.plot(actual.index, actual, color=INK, lw=1.4, ls="--", marker="o", ms=3,
                label="actual (held out)")
        ax.axvline(history.index[-1], color=INK_2, lw=0.8)
        ax.tick_params(axis="x", rotation=30)
    c = MODEL_COLOR["SARIMAX"]
    for lvl, alpha in zip(sorted(sarimax_bands, reverse=True), (0.15, 0.3, 0.45)):
        b = sarimax_bands[lvl]
        axes[0].fill_between(b.index, b["lo"], b["hi"], color=c, alpha=alpha, lw=0,
                             label=f"{lvl:.0%} PI")
    axes[0].plot(sarimax_mean.index, sarimax_mean, color=c, lw=2, label="SARIMAX mean")
    if conformal is not None:
        for col in ("lo", "hi"):
            axes[0].plot(conformal.index, conformal[col], color=c, lw=1.2, ls=":",
                         label="calibrated 90% PI (conformal)" if col == "lo" else None)
    axes[0].set_title(f"SARIMAX fan chart, origin {history.index[-1].date()}")
    axes[0].set_ylabel("GWh/day")
    axes[0].legend(loc="lower left", fontsize=8)
    c = MODEL_COLOR["LightGBM"]
    axes[1].fill_between(lgbm.index, lgbm["lo"], lgbm["hi"], color=c, alpha=0.25, lw=0,
                         label="90% quantile band")
    axes[1].plot(lgbm.index, lgbm["mean"], color=c, lw=2, label="LightGBM mean")
    axes[1].set_title("LightGBM, same origin")
    axes[1].legend(loc="lower left", fontsize=8)
    _save(fig, path)


def feature_importance(imp: pd.Series, path: Path, top: int = 15) -> None:
    """Horizontal bar chart of LightGBM gain importance (share of total)."""
    s = (imp / imp.sum()).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(s.index, 100 * s.values, color=MODEL_COLOR["LightGBM"], height=0.7)
    for i, v in enumerate(s.values):
        ax.text(100 * v + 0.3, i, f"{100 * v:.1f}%", va="center", fontsize=8, color=INK_2)
    ax.set_xlabel("share of total split gain (%)")
    ax.set_title("LightGBM feature importance (gain), last backtest fold")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def calibration_by_horizon(cov: pd.DataFrame, level: float, path: Path) -> None:
    """Coverage and mean width of 90% intervals by horizon, raw vs conformal.

    ``cov`` has a MultiIndex (model, method, h) and columns coverage, width.
    """
    cov = cov.sort_index()
    models = [m for m in ["SARIMAX", "LightGBM", "Combination (inv_mse)"]
              if m in cov.index.get_level_values(0)]
    styles = {"raw": dict(ls="--", marker="o", alpha=0.75),
              "conformal (scaled)": dict(ls="-", marker="s", alpha=1.0)}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for model in models:
        c = MODEL_COLOR[model]
        for method, st in styles.items():
            if (model, method) not in cov.index.droplevel(2):
                continue
            g = cov.loc[(model, method)]
            lab = f"{model}, {method}"
            axes[0].plot(g.index, 100 * g["coverage"], color=c, lw=1.8, ms=3.5, label=lab, **st)
            axes[1].plot(g.index, g["width"], color=c, lw=1.8, ms=3.5, label=lab, **st)
    axes[0].axhline(100 * level, color=INK, lw=1, ls="--")
    axes[0].annotate(f"nominal {level:.0%}", (14, 100 * level), xytext=(0, 4),
                     textcoords="offset points", ha="right", color=INK_2, fontsize=8.5)
    axes[0].set_title("Coverage of 90% intervals, test origins 2021-2022")
    axes[0].set_xlabel("horizon (days ahead)")
    axes[0].set_ylabel("coverage (%)")
    axes[0].legend(loc="lower left", fontsize=7.5)
    axes[1].set_title("Mean interval width")
    axes[1].set_xlabel("horizon (days ahead)")
    axes[1].set_ylabel("GWh/day")
    axes[1].set_ylim(bottom=0)
    _save(fig, path)
