"""Evaluation helpers: Diebold-Mariano tables, probabilistic scores, regime breakdowns."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.conformal import TAUS, qcol
from src.metrics import (
    coverage, crps_from_quantiles, diebold_mariano, dm_on_loss_differential, interval_score,
    mae, mape, pinball_loss, rmse,
)

# Regimes by target date. The 2022 gas-price crisis started with the Russian invasion of
# Ukraine (24 Feb 2022); the visible demand reduction at given temperature starts in
# autumn 2022 (see the demand-temperature scatter).
REGIMES: list[tuple[str, str, str]] = [
    ("2021 pre-crisis", "2021-01-01", "2021-12-31"),
    ("2022 crisis", "2022-01-01", "2022-12-31"),
    ("  2022 Jan-Aug", "2022-01-01", "2022-08-31"),
    ("  2022 Sep-Dec (demand reduction)", "2022-09-01", "2022-12-31"),
]


def dm_table(
    bt: pd.DataFrame, a: str, b: str, horizons: tuple[int, ...] = (1, 3, 7, 14)
) -> pd.DataFrame:
    """DM tests of model ``a`` vs ``b``: selected horizons and the 1-14 day pooled loss.

    Negative statistic = ``a`` has the lower loss. Per horizon there is one forecast per
    origin; origins are 21 days apart so the h-step errors are not overlapping.
    """
    wide = bt[bt["model"].isin([a, b])].pivot_table(
        index=["origin", "h"], columns="model", values="error")
    rows = []
    for h in horizons:
        e = wide.xs(h, level="h")
        for power, loss in [(2, "squared"), (1, "absolute")]:
            stat, p = diebold_mariano(e[a].to_numpy(), e[b].to_numpy(), h=1, power=power)
            rows.append({"comparison": f"{a} vs {b}", "horizon": str(h), "loss": loss,
                         "DM_stat": stat, "p_value": p, "n": len(e)})
    for power, loss in [(2, "squared"), (1, "absolute")]:
        la = (wide[a].abs() ** power).groupby(level="origin").mean()
        lb = (wide[b].abs() ** power).groupby(level="origin").mean()
        stat, p = dm_on_loss_differential((la - lb).to_numpy())
        rows.append({"comparison": f"{a} vs {b}", "horizon": "1-14 pooled", "loss": loss,
                     "DM_stat": stat, "p_value": p, "n": len(la)})
    return pd.DataFrame(rows).set_index(["comparison", "horizon", "loss"])


def dm_daily(da: pd.DataFrame, a: str, b: str, max_lag: int = 7) -> pd.DataFrame:
    """DM tests on daily day-ahead errors (Newey-West long-run variance)."""
    wide = da[da["model"].isin([a, b])].pivot(index="date", columns="model", values="error")
    rows = []
    for power, loss in [(2, "squared"), (1, "absolute")]:
        stat, p = diebold_mariano(wide[a].to_numpy(), wide[b].to_numpy(), h=1, power=power,
                                  max_lag=max_lag)
        rows.append({"comparison": f"{a} vs {b}", "loss": loss, "DM_stat": stat,
                     "p_value": p, "n": len(wide)})
    return pd.DataFrame(rows).set_index(["comparison", "loss"])


def probabilistic_scores(fc: pd.DataFrame, level: float, taus: np.ndarray = TAUS) -> pd.Series:
    """Coverage, width, interval score, tail pinball losses and quantile-CRPS of one method."""
    alpha = 1 - level
    q = fc[[qcol(t) for t in taus]].to_numpy()
    return pd.Series({
        "coverage": coverage(fc["y"], fc["lo"], fc["hi"]),
        "below_lo": float(np.mean(fc["y"] < fc["lo"])),
        "above_hi": float(np.mean(fc["y"] > fc["hi"])),
        "width": float(np.mean(fc["hi"] - fc["lo"])),
        "interval_score": interval_score(fc["y"], fc["lo"], fc["hi"], level),
        f"pinball_{alpha / 2:.2f}": pinball_loss(fc["y"], fc["lo"], alpha / 2),
        f"pinball_{1 - alpha / 2:.2f}": pinball_loss(fc["y"], fc["hi"], 1 - alpha / 2),
        "CRPS_q19": crps_from_quantiles(fc["y"], q, taus),
        "n": len(fc),
    })


def regime_label(dates: pd.Series) -> pd.Series:
    """Top-level regime of each target date (2021 vs 2022)."""
    return np.where(pd.DatetimeIndex(dates).year == 2021, "2021 pre-crisis", "2022 crisis")


def regime_table(fc: pd.DataFrame, extra: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Point-forecast errors (and interval coverage, if ``lo``/``hi`` exist) by regime.

    ``fc`` is a long table with model, date, y, mean, error, optional lo/hi.
    ``extra`` maps a label to a calibrated table whose coverage is added as a column.
    """
    rows = []
    for name, start, end in REGIMES:
        sub = fc[(fc["date"] >= start) & (fc["date"] <= end)]
        for model, g in sub.groupby("model", sort=False):
            rec = {"regime": name, "model": model, "MAE": mae(g["y"], g["mean"]),
                   "RMSE": rmse(g["y"], g["mean"]), "MAPE": mape(g["y"], g["mean"]),
                   "bias": float(g["error"].mean()), "n": len(g)}
            if {"lo", "hi"} <= set(g.columns) and g["lo"].notna().all():
                rec["coverage_raw"] = coverage(g["y"], g["lo"], g["hi"])
            rows.append(rec)
        for label, cal in (extra or {}).items():
            c = cal[(cal["date"] >= start) & (cal["date"] <= end)]
            for model, g in c.groupby("model", sort=False):
                for r in rows:
                    if r["regime"] == name and r["model"] == model:
                        r[f"coverage_{label}"] = coverage(g["y"], g["lo"], g["hi"])
                        r[f"width_{label}"] = float(np.mean(g["hi"] - g["lo"]))
    return pd.DataFrame(rows).set_index(["regime", "model"])
