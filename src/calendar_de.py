"""German public-holiday and calendar helpers (no external dependency)."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd


def easter_sunday(year: int) -> dt.date:
    """Gregorian Easter Sunday (anonymous Gregorian / Meeus-Jones-Butcher algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = ((h + ell - 7 * m + 114) % 31) + 1
    return dt.date(year, month, day)


def german_national_holidays(year: int) -> dict[dt.date, str]:
    """Nation-wide German public holidays (regional ones are ignored).

    2017 additionally had a one-off national holiday on 31 October (Reformation 500).
    """
    es = easter_sunday(year)
    days = {
        dt.date(year, 1, 1): "New Year",
        es - dt.timedelta(days=2): "Good Friday",
        es + dt.timedelta(days=1): "Easter Monday",
        dt.date(year, 5, 1): "Labour Day",
        es + dt.timedelta(days=39): "Ascension",
        es + dt.timedelta(days=50): "Whit Monday",
        dt.date(year, 10, 3): "German Unity",
        dt.date(year, 12, 25): "Christmas Day",
        dt.date(year, 12, 26): "Boxing Day",
    }
    if year == 2017:
        days[dt.date(2017, 10, 31)] = "Reformation 500"
    return days


def calendar_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic calendar features for the given dates.

    These are known in advance for any future date, so they never leak information.
    """
    years = range(index.year.min(), index.year.max() + 1)
    hol = {d for y in years for d in german_national_holidays(y)}
    dates = index.date
    is_hol = np.array([d in hol for d in dates], dtype=int)
    # Bridge/Christmas period: 24.12 - 31.12 is low-demand for industry.
    xmas = ((index.month == 12) & (index.day >= 24)).astype(int)
    doy = index.dayofyear.to_numpy()
    return pd.DataFrame(
        {
            "dow": index.dayofweek,
            "is_weekend": (index.dayofweek >= 5).astype(int),
            "is_holiday": is_hol,
            "xmas_period": xmas,
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        },
        index=index,
    )
