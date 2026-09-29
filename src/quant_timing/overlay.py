from __future__ import annotations

import pandas as pd

FUTURES_COLUMNS = {"FUTURES", "IF", "IC", "IM"}

# Liquid CFFEX underlyings. Sleeves that are not listed fall back to IF.
DEFAULT_HEDGE = {
    "hs300": "IF",
    "csi300_value": "IF",
    "csi300_growth": "IF",
    "cni_value": "IF",
    "cni_growth": "IF",
    "dividend": "IF",
    "cyclical": "IF",
    "bank": "IF",
    "broker": "IF",
    "pharma": "IF",
    "electronics": "IF",
    "csi500": "IC",
    "csi1000": "IM",
}


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
    betas: pd.DataFrame | None = None,
    fallback_betas: pd.DataFrame | None = None,
    hedge_map: dict[str, str] | None = None,
    available: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fund stocks and margin from capital, and set net beta with futures.

    Futures notionals are marked to market and are not treated as cash. A short futures
    position therefore does not create an extra bond holding. Each sleeve is hedged with
    its own contract when that contract has a price; otherwise the hedge falls back to IF.
    """
    if not 0.0 <= margin_rate < 1.0:
        raise ValueError("margin_rate must be in [0, 1)")
    equity_columns = [column for column in weights.columns if column not in FUTURES_COLUMNS | {"CASH", "MARGIN"}]
    equity = weights[equity_columns].astype(float)
    total = equity.sum(axis=1)
    active = total.gt(1e-12)
    mix = equity.div(total.where(active), axis=0).fillna(0.0)
    contracts = ["IF", "IC", "IM"]
    out = pd.DataFrame(0.0, index=weights.index, columns=[*equity_columns, *contracts, "CASH", "MARGIN"])
    mapping = {**DEFAULT_HEDGE, **(hedge_map or {})}
    beta_frame = _betas(mix, betas)
    fallback_frame = _betas(mix, fallback_betas)
    listed = _available(weights.index, available)
    for stamp in weights.index:
        if not bool(active.loc[stamp]):
            out.loc[stamp, "CASH"] = 1.0
            continue
        target = float(scale.loc[stamp])
        exposures = _contract_exposure(
            mix.loc[stamp],
            beta_frame.loc[stamp],
            fallback_frame.loc[stamp],
            mapping,
            listed.loc[stamp],
        )
        beta_sum = sum(exposures.values())
        if beta_sum <= 1e-12 or target >= 1.0 - 1e-12:
            out.loc[stamp, equity_columns] = mix.loc[stamp]
            continue
        stock = (1.0 + margin_rate * beta_sum * target) / (1.0 + margin_rate * beta_sum)
        hedge = stock - target
        margin = margin_rate * beta_sum * hedge
        out.loc[stamp, equity_columns] = mix.loc[stamp] * stock
        out.loc[stamp, "MARGIN"] = margin
        out.loc[stamp, "CASH"] = 1.0 - stock - margin
        for contract, exposure in exposures.items():
            out.loc[stamp, contract] = -exposure * hedge
    out.index.name = weights.index.name
    return out


def _betas(mix: pd.DataFrame, betas: pd.DataFrame | None) -> pd.DataFrame:
    if betas is None:
        return pd.DataFrame(1.0, index=mix.index, columns=mix.columns)
    aligned = betas.reindex(index=mix.index, columns=mix.columns)
    return aligned.fillna(1.0)


def _available(index: pd.Index, available: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["IF", "IC", "IM"]
    if available is None:
        return pd.DataFrame(True, index=index, columns=columns)
    aligned = available.reindex(index=index, columns=columns)
    return aligned.fillna(False).astype(bool)


def _contract_exposure(
    mix: pd.Series,
    beta: pd.Series,
    fallback_beta: pd.Series,
    mapping: dict[str, str],
    listed: pd.Series,
) -> dict[str, float]:
    exposures = {"IF": 0.0, "IC": 0.0, "IM": 0.0}
    for sleeve, weight in mix.items():
        weight_value = float(weight)
        if weight_value <= 0.0:
            continue
        contract = mapping.get(str(sleeve), "IF")
        if contract not in exposures or not bool(listed.get(contract, False)):
            exposures["IF"] += weight_value * float(fallback_beta.get(sleeve, 1.0))
            continue
        exposures[contract] += weight_value * float(beta.get(sleeve, 1.0))
    return exposures
