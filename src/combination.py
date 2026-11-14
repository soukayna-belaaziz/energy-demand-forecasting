"""Forecast combination of SARIMAX and LightGBM.

Two schemes:

* ``equal``: simple average (the "forecast combination puzzle": hard to beat in practice).
* ``inv_mse``: weights proportional to 1 / MSE of each member at the same horizon,
  estimated from past errors whose target date is <= the origin (expanding window,
  calibration folds included). With fewer than ``min_obs`` past errors at a horizon
  the scheme falls back to equal weights.

The combined interval uses the weighted average of the members' implied standard
deviations (a rough, uncalibrated band that is recalibrated by ``src.conformal``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.conformal import implied_sd


def inverse_mse_weights(errors: pd.DataFrame, min_obs: int = 10) -> pd.Series:
    """Weights proportional to 1/MSE per column of a past-errors frame; equal if too few rows."""
    errors = errors.dropna()
    k = errors.shape[1]
    if len(errors) < min_obs:
        return pd.Series(1.0 / k, index=errors.columns)
    inv = 1.0 / (errors**2).mean()
    return inv / inv.sum()


def combine_backtest(
    bt: pd.DataFrame,
    members: list[str],
    level: float,
    scheme: str = "equal",
    min_obs: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine member forecasts origin by origin.

    ``bt`` is the long backtest table (several models). Returns the combined long table
    (same columns as ``bt`` for the model ``"Combination (<scheme>)"``) and a table of
    the weights used at every (origin, h).
    """
    if scheme not in {"equal", "inv_mse"}:
        raise ValueError("scheme must be 'equal' or 'inv_mse'")
    sub = bt[bt["model"].isin(members)].copy()
    sub["sd"] = implied_sd(sub["lo"], sub["hi"], level)
    keys = ["origin", "date", "h"]
    mean = sub.pivot_table(index=keys, columns="model", values="mean")[members]
    sd = sub.pivot_table(index=keys, columns="model", values="sd")[members]
    err = sub.pivot_table(index=keys, columns="model", values="error")[members]
    first = sub.drop_duplicates(keys).set_index(keys)[["y", "scale"]].loc[mean.index]
    dates = err.index.get_level_values("date")

    w_rows = []
    for origin in mean.index.get_level_values("origin").unique():
        past = err[dates <= origin]
        for h in range(1, int(mean.index.get_level_values("h").max()) + 1):
            if scheme == "equal":
                w = pd.Series(1.0 / len(members), index=members)
            else:
                past_h = past[past.index.get_level_values("h") == h]
                w = inverse_mse_weights(past_h, min_obs)
            w_rows.append({"origin": origin, "h": h, **w.to_dict()})
    weights = pd.DataFrame(w_rows)
    wfull = mean.reset_index()[keys].merge(weights, on=["origin", "h"], how="left")
    wmat = wfull[members].to_numpy()
    comb_mean = np.sum(mean.to_numpy() * wmat, axis=1)
    comb_sd = np.sum(sd.to_numpy() * wmat, axis=1)
    z = stats.norm.ppf(0.5 + level / 2)
    out = pd.DataFrame({
        "model": f"Combination ({scheme})",
        "origin": wfull["origin"], "date": wfull["date"], "h": wfull["h"],
        "y": first["y"].to_numpy(), "mean": comb_mean,
        "lo": comb_mean - z * comb_sd, "hi": comb_mean + z * comb_sd,
        "scale": first["scale"].to_numpy(),
    })
    out["error"] = out["y"] - out["mean"]
    return out, weights


def combine_daily(
    da: pd.DataFrame, members: list[str], scheme: str = "equal", min_obs: int = 28
) -> pd.DataFrame:
    """Combine day-ahead forecasts; inverse-MSE weights use all errors dated before each day."""
    mean = da.pivot(index="date", columns="model", values="mean")[members]
    err = da.pivot(index="date", columns="model", values="error")[members]
    y = da.drop_duplicates("date").set_index("date")["y"].loc[mean.index]
    out_mean = []
    for d in mean.index:
        if scheme == "equal":
            w = np.full(len(members), 1.0 / len(members))
        else:
            w = inverse_mse_weights(err[err.index < d], min_obs).to_numpy()
        out_mean.append(float(mean.loc[d].to_numpy() @ w))
    out = pd.DataFrame({"date": mean.index, "mean": out_mean, "y": y.to_numpy(),
                        "lo": np.nan, "hi": np.nan, "model": f"Combination ({scheme})"})
    out["error"] = out["y"] - out["mean"]
    return out
