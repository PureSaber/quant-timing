from __future__ import annotations

import pandas as pd

from quant_timing.style import style_sleeves

_FUTURES_COLUMNS = {"FUTURES", "IF", "IC", "IM"}


def assemble_weights(
    prices: pd.DataFrame,
    position: pd.DataFrame,
    style: dict | None,
    internal: pd.DataFrame | None,
    market: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the equity budget across the market or ready style sleeves, with cash as residual."""
    if not position.index.equals(prices.index):
        raise ValueError("position history does not match the price index")
    sleeves = style_sleeves(style) if style else []
    columns = [market, *sleeves, "CASH"]
    weights = pd.DataFrame(0.0, index=prices.index, columns=columns)
    matched = pd.DataFrame(0.0, index=prices.index, columns=columns)
    scale = position["position_scale"].astype(float)
    if internal is None:
        weights[market] = scale
        matched[market] = scale
    else:
        if style is None:
            raise ValueError("style weights require a style configuration")
        if not internal.index.equals(prices.index):
            raise ValueError("style weights do not match the price index")
        ready = internal.notna().all(axis=1)
        weights.loc[~ready, market] = scale.loc[~ready].to_numpy()
        matched.loc[~ready, market] = scale.loc[~ready].to_numpy()
        for sleeve in sleeves:
            sleeve_weight = scale.loc[ready] * internal.loc[ready, sleeve].astype(float)
            weights.loc[ready, sleeve] = sleeve_weight.to_numpy()
        for pair in style["pairs"]:
            share = float(pair["group_weight"]) * 0.5
            matched.loc[ready, pair["left"]] = (scale.loc[ready] * share).to_numpy()
            matched.loc[ready, pair["right"]] = (scale.loc[ready] * share).to_numpy()
        for group in style.get("groups") or []:
            share = float(group["group_weight"]) / len(group["members"])
            for member in group["members"]:
                matched.loc[ready, member] = (scale.loc[ready] * share).to_numpy()
    weights["CASH"] = 1.0 - scale
    matched["CASH"] = 1.0 - scale
    _assert_book(weights)
    _assert_book(matched)
    return weights, matched


FUTURES_COLUMNS = {"FUTURES", "IF", "IC", "IM"}


def _assert_book(weights: pd.DataFrame) -> None:
    funded = [column for column in weights.columns if column not in _FUTURES_COLUMNS]
    totals = weights[funded].sum(axis=1)
    if not totals.sub(1.0).abs().le(1e-8).all():
        raise ValueError("funded portfolio weights must sum to 1")
    long_only = [column for column in funded]
    if weights[long_only].lt(-1e-12).any().any():
        raise ValueError("funded weights must be long-only")


def simulate(
    weights: pd.DataFrame,
    matched: pd.DataFrame,
    prices: pd.DataFrame,
    market: str,
    cost_bps: float,
    cash_returns: pd.Series | None = None,
    impact_coef: float = 0.0,
    extra_returns: dict[str, pd.Series] | None = None,
    adv: pd.Series | None = None,
    capital: float | None = None,
) -> pd.DataFrame:
    """Earn the next bar's return with today's weights and charge today's turnover on that bar.

    The first observation is the funding date: its turnover is zero and it has no realized return.
    """
    asset = prices.pct_change()
    asset["CASH"] = 0.0
    if cash_returns is not None:
        asset["CASH"] = cash_returns.reindex(asset.index)
    for name, series in (extra_returns or {}).items():
        asset[name] = series.reindex(asset.index)
    if "MARGIN" in weights.columns:
        asset["MARGIN"] = 0.0
    gross, net, cost, turnover = _path(weights, asset, cost_bps, impact_coef, adv, capital)
    matched_gross, matched_net, matched_cost, matched_turnover = _path(
        matched, asset, cost_bps, impact_coef, adv, capital
    )
    frame = pd.DataFrame(
        {
            "decision_date": pd.Series(weights.index, index=weights.index).shift(1),
            "gross_return": gross,
            "net_return": net,
            "cost": cost,
            "turnover": turnover,
            "benchmark_return": prices[market].pct_change(),
            "matched_gross_return": matched_gross,
            "matched_net_return": matched_net,
            "matched_cost": matched_cost,
            "matched_turnover": matched_turnover,
        },
        index=weights.index,
    )
    frame.index.name = "date"
    frame["nav"] = _nav(frame["net_return"])
    return frame


def _path(
    weights: pd.DataFrame,
    asset: pd.DataFrame,
    cost_bps: float,
    impact_coef: float = 0.0,
    adv: pd.Series | None = None,
    capital: float | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    columns = list(weights.columns)
    returns = asset.reindex(columns=columns)
    if "CASH" in returns.columns:
        returns["CASH"] = returns["CASH"].fillna(0.0)
    if "MARGIN" in returns.columns:
        returns["MARGIN"] = returns["MARGIN"].fillna(0.0)
    gross_parts = weights.shift(1) * returns
    for column in _FUTURES_COLUMNS:
        if column not in gross_parts.columns:
            continue
        idle = weights[column].shift(1).abs().fillna(0.0).le(1e-12)
        gross_parts.loc[idle, column] = gross_parts.loc[idle, column].fillna(0.0)
    gross = gross_parts.sum(axis=1, min_count=len(columns))
    traded = weights.diff().abs().sum(axis=1) / 2.0
    traded.iloc[0] = 0.0
    turnover = traded.shift(1)
    cost = turnover * (cost_bps / 10_000.0)
    if impact_coef:
        cost = cost + _impact(turnover, float(impact_coef), adv, capital)
    return gross, gross - cost, cost, turnover


def _impact(
    turnover: pd.Series,
    impact_coef: float,
    adv: pd.Series | None,
    capital: float | None,
) -> pd.Series:
    traded = turnover.clip(lower=0.0)
    if adv is None or capital is None:
        return impact_coef * traded.pow(1.5)
    activity = adv.reindex(turnover.index).replace(0, pd.NA)
    participation = traded * float(capital) / activity
    return impact_coef * traded * participation.clip(lower=0.0).pow(0.5)


def _nav(net_return: pd.Series) -> pd.Series:
    nav = pd.Series(1.0, index=net_return.index, dtype=float)
    for loc in range(1, len(nav)):
        value = net_return.iloc[loc]
        previous = float(nav.iloc[loc - 1])
        nav.iloc[loc] = previous if pd.isna(value) else previous * (1.0 + float(value))
    return nav
