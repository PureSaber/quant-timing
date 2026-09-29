from __future__ import annotations

import numpy as np
import pandas as pd


def constrain_scale(
    scale: pd.Series,
    equity_return: pd.Series,
    cash_return: pd.Series,
    *,
    max_daily_change: float | None,
    drawdown_line: float | None,
    drawdown_floor: float,
) -> pd.Series:
    """Limit day-to-day changes, then let a drawdown line cut exposure immediately.

    The first finite scale is the initial allocation and is not slowed down. The net asset
    value used for the line only includes returns already realized before the decision.
    """
    values = scale.astype(float).to_numpy()
    equity = equity_return.reindex(scale.index).to_numpy(dtype=float)
    cash = cash_return.reindex(scale.index).fillna(0.0).to_numpy(dtype=float)
    out = np.zeros(len(values), dtype=float)
    nav = 1.0
    peak = 1.0
    previous: float | None = None
    for loc, raw in enumerate(values):
        proposed = 0.0 if not np.isfinite(raw) else float(raw)
        if previous is not None and max_daily_change is not None:
            step = float(max_daily_change)
            proposed = previous + min(max(proposed - previous, -step), step)
        if drawdown_line is not None and peak > 0 and nav / peak - 1.0 <= -float(drawdown_line):
            proposed = min(proposed, float(drawdown_floor))
        out[loc] = proposed
        if np.isfinite(raw):
            previous = proposed
        nxt = loc + 1
        if nxt < len(values) and np.isfinite(equity[nxt]):
            realized = proposed * float(equity[nxt]) + (1.0 - proposed) * float(cash[nxt])
            nav *= 1.0 + realized
            peak = max(peak, nav)
    return pd.Series(out, index=scale.index, name="position_scale")


def full_equity_return(
    prices: pd.DataFrame,
    market: str,
    internal: pd.DataFrame | None,
) -> pd.Series:
    """Return of a fully invested sleeve mix. The position scale is applied later."""
    returns = prices.pct_change()
    if internal is None:
        return returns[market]
    sleeves = list(internal.columns)
    mix = (internal.fillna(0.0) * returns[sleeves]).sum(axis=1, min_count=len(sleeves))
    ready = internal.notna().all(axis=1)
    equity = returns[market].copy()
    equity.loc[ready] = mix.loc[ready]
    return equity


def apply_earnings_yield_cap(
    scale: pd.Series,
    peg: pd.Series,
    *,
    lookback: int,
    expensive_percentile: float,
    scale_cap: float,
) -> pd.Series:
    """Cap exposure when the earnings yield is cheap versus its own past.

    The percentile uses only observations through the decision date. A low percentile
    means the published valuation is expensive.
    """
    earnings_yield = 1.0 / peg.astype(float).replace(0, np.nan)
    percentile = earnings_yield.rolling(lookback).apply(_last_percent_rank, raw=True)
    capped = scale.astype(float).copy()
    expensive = percentile.notna() & percentile.lt(expensive_percentile)
    capped.loc[expensive] = np.minimum(capped.loc[expensive].to_numpy(dtype=float), float(scale_cap))
    return capped


def _last_percent_rank(values: np.ndarray) -> float:
    if np.isnan(values).any():
        return float("nan")
    return float(pd.Series(values).rank(pct=True).iloc[-1])
