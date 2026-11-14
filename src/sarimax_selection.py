"""Choose the SARIMAX specification on the pre-test validation folds only.

The baseline model (ARIMA(1,1,2) errors, weighted HDD, holiday and weekday dummies)
leaves some residual autocorrelation (Ljung-Box p = 0.003-0.047). Candidate fixes,
each motivated by a residual pattern or by the physics of heating demand:

* ``+ weekly AR``: a seasonal AR(1) at lag 7 on the ARMA errors (stochastic weekly
  persistence on top of the deterministic weekday dummies). Weekly Fourier terms are
  not a separate candidate: with daily data, six weekday dummies already span the full
  weekly Fourier basis (K = 3), so Fourier terms would only restrict them.
* ``+ non-linear HDD``: a second hinge ``max(5 - T_w, 0)`` (the heating response steepens
  in frost) and ``HDD x weekend`` (weekend heating slope differs: offices and industry).
* both together.

Rule, fixed in advance: the lowest pooled 1-14 day MAE over the 22 validation origins
2019-09-29 .. 2020-12-13 (all targets before the first test origin). Ljung-Box tests
on the fit at the first test origin are reported as a diagnostic.
"""
from __future__ import annotations

import pandas as pd

from src.backtest import Fold, run_backtest
from src.diagnostics import ljung_box
from src.metrics import score_table
from src.models import SarimaxForecaster, SarimaxSpec


def candidate_specs(order: tuple[int, int, int]) -> list[SarimaxSpec]:
    """Baseline and the three candidate fixes, all with the BIC-selected ARIMA order."""
    weekly_ar = (1, 0, 0, 7)
    return [
        SarimaxSpec("baseline", order),
        SarimaxSpec("+ weekly AR", order, seasonal_order=weekly_ar),
        SarimaxSpec("+ non-linear HDD", order, hdd_knots=(5.0,), hdd_weekend=True),
        SarimaxSpec("+ non-linear HDD + weekly AR", order, seasonal_order=weekly_ar,
                    hdd_knots=(5.0,), hdd_weekend=True),
    ]


def select_sarimax(
    y: pd.Series,
    temp: pd.Series,
    archive: pd.DataFrame,
    specs: list[SarimaxSpec],
    valid_folds: list[Fold],
    first_test_fold: Fold,
    level: float,
) -> tuple[SarimaxSpec, pd.DataFrame, dict[str, SarimaxForecaster], pd.DataFrame]:
    """Run every candidate on the validation folds and pick the lowest-MAE one.

    Returns the chosen spec, the comparison table, the fitted forecasters (their
    warm-start state continues into the test folds) and the validation forecasts.
    """
    rows, fcs, parts = [], {}, []
    init = None
    for spec in specs:
        fc = SarimaxForecaster(y, temp, archive, spec, level=level, init_params=init)
        bt = run_backtest(y, valid_folds, {spec.name: fc})
        if init is None:  # seed later candidates with the baseline's first-fold estimates
            init = fc.results[valid_folds[0].origin].params
        sc = score_table(bt, ["model"]).iloc[0]
        fc(first_test_fold)  # fit on all pre-test data, for residual diagnostics
        res = fc.results[first_test_fold.origin]
        lb = ljung_box(res.resid.iloc[8:], [7, 14, 28], model_df=spec.n_arma)
        rows.append({
            "spec": spec.name, "valid_MAE": sc["MAE"], "valid_RMSE": sc["RMSE"],
            "valid_MAPE": sc["MAPE"], "valid_coverage": sc["coverage"],
            "BIC_pretest": res.bic,
            "LB_p_lag7": lb.loc[7, "lb_pvalue"], "LB_p_lag14": lb.loc[14, "lb_pvalue"],
            "LB_p_lag28": lb.loc[28, "lb_pvalue"],
        })
        fcs[spec.name] = fc
        parts.append(bt)
    table = pd.DataFrame(rows).set_index("spec")
    chosen = next(s for s in specs if s.name == table["valid_MAE"].idxmin())
    return chosen, table, fcs, pd.concat(parts, ignore_index=True)
