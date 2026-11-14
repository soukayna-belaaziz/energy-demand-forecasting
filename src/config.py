"""Central configuration: paths, sample period, backtest design, seeds."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
TABLE_DIR = ROOT / "reports" / "tables"

SEED = 42

# Sample: both German market areas (NCG + GASPOOL) published data from
# 2016-10-01; the ERA5 temperature extract ends on 2022-12-31.
START = "2016-10-01"
END = "2022-12-31"

# Heating threshold (deg C) for heating degree days.
HDD_BASE = 15.0
SEASONAL_PERIOD = 7


@dataclass(frozen=True)
class BacktestConfig:
    """Rolling-origin (expanding window) backtest design."""

    first_origin: str = "2021-01-03"  # last training day of the first fold
    step_days: int = 21  # spacing between consecutive origins
    horizon: int = 14  # forecast days 1..horizon
    interval_level: float = 0.90  # nominal prediction-interval coverage


@dataclass(frozen=True)
class TempForecastConfig:
    """Assumed temperature-forecast error, used to avoid using perfect future weather.

    The forecast for day t+h issued at t is the realised ERA5 value plus Gaussian
    noise with standard deviation ``sd_1 + sd_slope * (h - 1)`` (deg C). These are
    stated assumptions, not estimates from a real forecast archive.
    """

    sd_1: float = 0.8
    sd_slope: float = 0.2
