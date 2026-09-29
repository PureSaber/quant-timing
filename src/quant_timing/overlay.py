from __future__ import annotations

import pandas as pd


def cash_returns_from_yield(index: pd.Index, annual_yield: pd.Series, daycount: int = 252) -> pd.Series:
    """Daily cash return from an annual yield published on or before that date.

    Dates before the first print earn zero. The series is not back-filled from the future.
    """
    if daycount <= 0:
        raise ValueError("daycount must be positive")
    aligned = annual_yield.sort_index().reindex(index, method="ffill")
    daily = aligned / float(daycount)
    return daily.fillna(0.0)


def apply_futures_overlay(
    weights: pd.DataFrame,
    scale: pd.Series,
    *,
    margin_rate: float,
) -> pd.DataFrame:
    """Keep a fully invested stock book and use futures to set net equity beta.

    Column sum stays 1. Futures may be short. Cash earns a yield on the uninvested and
    hedged residual; margin earns zero.
    """
    if not 0.0 <= margin_rate < 1.0:
        raise ValueError("margin_rate must be in [0, 1)")
    equity_columns = [column for column in weights.columns if column not in {"CASH", "FUTURES", "MARGIN"}]
    equity = weights[equity_columns]
    total = equity.sum(axis=1)
    active = total.gt(1e-12)
    mix = equity.div(total.where(active), axis=0).fillna(0.0)
    out = pd.DataFrame(0.0, index=weights.index, columns=[*equity_columns, "FUTURES", "CASH", "MARGIN"])
    out.loc[active, equity_columns] = mix.loc[active, equity_columns]
    hedge_cash = (1.0 - scale).clip(lower=0.0)
    margin = hedge_cash * float(margin_rate)
    out.loc[active, "FUTURES"] = (scale - 1.0).loc[active]
    out.loc[active, "MARGIN"] = margin.loc[active]
    out.loc[active, "CASH"] = (hedge_cash - margin).loc[active]
    out.loc[~active, "CASH"] = 1.0
    out.index.name = weights.index.name
    return out
