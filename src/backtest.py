"""Rolling-origin (expanding-window) backtest."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.metrics import seasonal_naive_scale


@dataclass(frozen=True)
class Fold:
    """One backtest fold: train on dates <= origin, evaluate origin+1 .. origin+horizon."""

    origin: pd.Timestamp
    train_end_pos: int  # positional index of the origin (inclusive)
    test_index: pd.DatetimeIndex


def rolling_origin_splits(
    index: pd.DatetimeIndex, first_origin: str | pd.Timestamp, step: int, horizon: int
) -> Iterator[Fold]:
    """Yield expanding-window folds; the last fold still has a full test horizon."""
    first = index.get_loc(pd.Timestamp(first_origin))
    last = len(index) - 1 - horizon
    if first > last:
        raise ValueError("first origin leaves no room for a full horizon")
    for pos in range(first, last + 1, step):
        yield Fold(
            origin=index[pos],
            train_end_pos=pos,
            test_index=index[pos + 1 : pos + 1 + horizon],
        )


# A forecaster gets the fold and returns DataFrame(index=test dates, cols mean/lo/hi)
Forecaster = Callable[[Fold], pd.DataFrame]


def run_backtest(
    y: pd.Series, folds: list[Fold], forecasters: dict[str, Forecaster], m: int = 7
) -> pd.DataFrame:
    """Run every forecaster on every fold; return a long table of forecasts and actuals."""
    rows = []
    for fold in folds:
        scale = seasonal_naive_scale(y.iloc[: fold.train_end_pos + 1].to_numpy(), m)
        actual = y.loc[fold.test_index]
        for name, fn in forecasters.items():
            fc = fn(fold)
            if not fc.index.equals(fold.test_index):
                raise ValueError(f"{name}: forecast index does not match the test window")
            rows.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "origin": fold.origin,
                        "date": fold.test_index,
                        "h": np.arange(1, len(fold.test_index) + 1),
                        "y": actual.to_numpy(),
                        "mean": fc["mean"].to_numpy(),
                        "lo": fc["lo"].to_numpy(),
                        "hi": fc["hi"].to_numpy(),
                        "scale": scale,
                    }
                )
            )
    out = pd.concat(rows, ignore_index=True)
    out["error"] = out["y"] - out["mean"]
    return out
