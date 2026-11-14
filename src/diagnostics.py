"""Exploratory diagnostics: decomposition, stationarity, ARIMA order selection, residual tests."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss


def mstl_decomposition(y: pd.Series, periods: tuple[int, ...] = (7, 365)) -> pd.DataFrame:
    """MSTL (multiple STL) with weekly and annual seasonality."""
    res = MSTL(y, periods=periods).fit()
    out = pd.DataFrame({"observed": y, "trend": res.trend, "resid": res.resid})
    seas = res.seasonal
    for p in periods:
        out[f"seasonal_{p}"] = seas[f"seasonal_{p}"]
    return out


def stationarity_tests(x: pd.Series) -> dict[str, float]:
    """ADF (H0: unit root) and KPSS (H0: level stationarity) p-values."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # KPSS p-values are truncated to [0.01, 0.1]
        return {
            "adf_p": float(adfuller(x.dropna(), autolag="AIC")[1]),
            "kpss_p": float(kpss(x.dropna(), regression="c", nlags="auto")[1]),
        }


def regression_residuals(y: pd.Series, x: pd.DataFrame) -> pd.Series:
    """OLS residuals of y on the exogenous regressors (the 'error' series ARIMA must model)."""
    return sm.OLS(y, sm.add_constant(x)).fit().resid


def ljung_box(resid: pd.Series, lags: list[int], model_df: int = 0) -> pd.DataFrame:
    """Ljung-Box Q statistics and p-values at the given lags."""
    lb = acorr_ljungbox(resid.dropna(), lags=lags, model_df=model_df)
    lb.index.name = "lag"
    return lb


def select_arima_order(
    y: pd.Series, x: pd.DataFrame, d: int = 1, max_p: int = 2, max_q: int = 2
) -> pd.DataFrame:
    """Grid search over ARIMA(p,d,q) errors for a regression model, ranked by BIC."""
    rows = []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = SARIMAX(y, exog=x, order=(p, d, q)).fit(disp=False, maxiter=300)
            lb = ljung_box(res.resid.iloc[d + 7 :], [7, 14, 28], model_df=p + q)
            rows.append(
                {
                    "p": p, "d": d, "q": q,
                    "aic": res.aic, "bic": res.bic,
                    "lb_p_lag7": lb.loc[7, "lb_pvalue"],
                    "lb_p_lag14": lb.loc[14, "lb_pvalue"],
                    "lb_p_lag28": lb.loc[28, "lb_pvalue"],
                    "converged": bool(res.mle_retvals.get("converged", False)),
                }
            )
    return pd.DataFrame(rows).sort_values("bic").reset_index(drop=True)


def acf_values(x: pd.Series, nlags: int = 28) -> np.ndarray:
    """Sample autocorrelations (convenience wrapper)."""
    return sm.tsa.acf(x.dropna(), nlags=nlags)
