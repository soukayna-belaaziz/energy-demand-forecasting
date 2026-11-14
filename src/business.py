"""Translate forecast errors into an illustrative gas balancing (imbalance) cost.

Stylised set-up, stated as assumptions:
* A shipper nominates (buys) the forecast volume for its portfolio day-ahead.
* If actual consumption is higher than the forecast (short position) the shortfall
  is settled at THE's positive balancing price instead of the VHP index; if it is
  lower (long) the surplus is cashed out at the negative balancing price.
* The cost of an error is therefore measured against the index price:
  short: e * (pos - index), long: |e| * (index - neg).
* The portfolio is a fixed share of German demand and inherits the model's
  relative errors. Real shippers have different (often noisier) portfolios and
  SLP volumes are allocated by profile, so this is a ranking device, not a P&L.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def imbalance_cost(
    errors: pd.DataFrame, prices: pd.DataFrame, share: float
) -> pd.DataFrame:
    """Per-forecast imbalance cost in EUR.

    ``errors`` needs columns date, error (actual - forecast, GWh/day) and model.
    """
    df = errors.merge(prices, left_on="date", right_index=True, how="inner")
    e_mwh = df["error"] * 1000.0 * share
    short_cost = np.clip(e_mwh, 0, None) * (df["pos_price"] - df["index_price"])
    long_cost = np.clip(-e_mwh, 0, None) * (df["index_price"] - df["neg_price"])
    df["cost_eur"] = short_cost + long_cost
    df["abs_error_mwh"] = e_mwh.abs()
    return df
