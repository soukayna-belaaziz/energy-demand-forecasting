"""Load and harmonise German daily natural-gas consumption and temperature.

Target: total daily gas consumption of the German market areas in GWh/day,
i.e. the sum of all published customer groups (SLP synthetic/analytic and
RLM with/without daily metering, H-gas and L-gas):

* 2016-10-01 .. 2021-09-30: NetConnect Germany (kWh) + GASPOOL (MWh)
* 2021-10-01 .. 2022-12-31: Trading Hub Europe (kWh), the merged market area
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config

NCG_COLS = [
    "HGasSLPsyn", "HGasSLPana", "LGasSLPsyn", "LGasSLPana",
    "HGasRLMmT", "LGasRLMmT", "HGasRLMoT", "LGasRLMoT",
]
SLP_PREFIX = "SLP"


def _read_ncg(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df["date"] = pd.to_datetime(df["DayOfUse"], format="%d.%m.%Y")
    out = pd.DataFrame(index=df["date"])
    out["slp"] = df[[c for c in NCG_COLS if "SLP" in c]].sum(axis=1).to_numpy() / 1e6
    out["rlm"] = df[[c for c in NCG_COLS if "RLM" in c]].sum(axis=1).to_numpy() / 1e6
    return out  # GWh


def _read_gaspool(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df["date"] = pd.to_datetime(df["Datum"], format="%d.%m.%Y")
    out = pd.DataFrame(index=df["date"])
    out["slp"] = df[[c for c in df.columns if c.startswith("SLP")]].sum(axis=1).to_numpy() / 1e3
    out["rlm"] = df[[c for c in df.columns if c.startswith("RLM")]].sum(axis=1).to_numpy() / 1e3
    return out  # GWh


def _read_the(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", thousands=",")
    df["date"] = pd.to_datetime(df["Gasday"], format="%d/%m/%Y")
    value_cols = [c for c in df.columns if c.startswith(("SLP", "RLM"))]
    df = df.dropna(subset=value_cols)  # last preliminary days lack RLM values
    out = pd.DataFrame(index=df["date"])
    out["slp"] = df[[c for c in value_cols if c.startswith("SLP")]].sum(axis=1).to_numpy() / 1e6
    out["rlm"] = df[[c for c in value_cols if c.startswith("RLM")]].sum(axis=1).to_numpy() / 1e6
    return out.sort_index()


def load_consumption(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """Daily German gas consumption (GWh/day) with SLP/RLM split and total."""
    ncg = _read_ncg(raw_dir / "ncg_consumption.csv")
    gp = _read_gaspool(raw_dir / "gaspool_consumption.csv")
    the = _read_the(raw_dir / "the_consumption.csv")
    pre = ncg.loc["2016-10-01":"2021-09-30"].add(gp.loc["2016-10-01":"2021-09-30"])
    post = the.loc["2021-10-01":]
    df = pd.concat([pre, post]).sort_index()
    df["total"] = df["slp"] + df["rlm"]
    df.index.name = "date"
    return df


def load_temperature(raw_dir: Path = config.RAW_DIR, country: str = "DE") -> pd.Series:
    """ERA5 country-average (NUTS0) daily 2 m air temperature from C3S, converted to deg C."""
    path = raw_dir / "era5_t2m_nuts0_daily.csv"
    with open(path, encoding="utf-8", errors="replace") as fh:
        skip = next(i for i, line in enumerate(fh) if line.startswith("Date,"))
    df = pd.read_csv(path, skiprows=skip, usecols=["Date", country], parse_dates=["Date"])
    s = df.set_index("Date")[country] - 273.15
    s.index.name = "date"
    return s.rename("temp")


def build_dataset(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """Merged, gap-checked daily dataset for the analysis period."""
    cons = load_consumption(raw_dir)
    temp = load_temperature(raw_dir)
    df = cons.join(temp, how="inner").loc[config.START : config.END]
    full = pd.date_range(config.START, config.END, freq="D")
    missing = full.difference(df.index)
    if len(missing):
        raise ValueError(f"{len(missing)} missing days, e.g. {missing[:5].tolist()}")
    df = df.asfreq("D")
    df["hdd"] = np.maximum(config.HDD_BASE - df["temp"], 0.0)
    return df


def load_imbalance_prices(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """THE daily balancing-energy prices (EUR/MWh), 2021-10-01 .. 2022-11-28."""
    df = pd.read_csv(raw_dir / "the_imbalance_prices.csv", sep=";", decimal=",",
                     na_values=["n/a"])
    df["date"] = pd.to_datetime(df["Gastag"], format="%d.%m.%Y")
    out = pd.DataFrame(
        {
            "pos_price": df["Positiver Ausgleichsenergiepreis (EUR/MWh)"].to_numpy(),
            "neg_price": df["Negativer Ausgleichsenergiepreis (EUR/MWh)"].to_numpy(),
            "index_price": df["VHP-Index (EUR/MWh)"].to_numpy(),
        },
        index=df["date"],
    ).sort_index()
    out.index.name = "date"
    return out
