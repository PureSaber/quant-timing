from __future__ import annotations

import pandas as pd


def style_sleeves(style: dict) -> list[str]:
    names: list[str] = []
    for pair in style.get("pairs") or []:
        names.extend((pair["left"], pair["right"]))
    for group in style.get("groups") or []:
        names.extend(group["members"])
    return names


def style_internal_weights(
    prices: pd.DataFrame,
    style: dict,
    signals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Long-only sleeve weights known at the close. Each ready row sums to 1."""
    window = int(style["return_window"])
    tilt = float(style["tilt"])
    threshold = float(style["threshold"])
    sleeves = style_sleeves(style)
    weights = pd.DataFrame(index=prices.index, columns=sleeves, dtype=float)
    ready = pd.Series(True, index=prices.index)
    for pair in style.get("pairs") or []:
        pair_ready, left_weight, right_weight = _pair_weights(
            prices, pair, signals, window, tilt, threshold
        )
        ready = ready & pair_ready
        weights[pair["left"]] = left_weight
        weights[pair["right"]] = right_weight
    for group in style.get("groups") or []:
        group_ready, member_weights = _group_weights(
            prices, group, signals, window, tilt, threshold
        )
        ready = ready & group_ready
        for member, member_weight in member_weights.items():
            weights[member] = member_weight
    weights.loc[~ready, :] = pd.NA
    weights = weights.astype(float)
    weights.index.name = "date"
    return weights


def _pair_weights(
    prices: pd.DataFrame,
    pair: dict,
    signals: pd.DataFrame | None,
    window: int,
    tilt: float,
    threshold: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    signal_name = str(pair.get("signal", "momentum"))
    spread, pair_ready = _spread(prices, pair, signals, window, signal_name)
    favor_left = signal_name != "crowding"
    left_share = _tilt_share(spread, pair_ready, tilt, threshold, favor_left=favor_left)
    group_weight = float(pair["group_weight"])
    left_weight = group_weight * left_share
    right_weight = group_weight - left_weight
    cap = pair.get("max_active_deviation")
    if cap is not None:
        benchmark = group_weight * 0.5
        left_weight = left_weight.clip(benchmark - float(cap), benchmark + float(cap))
        right_weight = group_weight - left_weight
    return pair_ready, left_weight, right_weight


def _group_weights(
    prices: pd.DataFrame,
    group: dict,
    signals: pd.DataFrame | None,
    window: int,
    tilt: float,
    threshold: float,
) -> tuple[pd.Series, dict[str, pd.Series]]:
    members = list(group["members"])
    signal_name = str(group.get("signal", "momentum"))
    scores = []
    ready = pd.Series(True, index=prices.index)
    for member in members:
        column = f"{group['name']}.{member}"
        if signal_name == "momentum":
            score = prices[member].pct_change(window)
        else:
            score = _signal_column(signals, column, prices.index)
            if signal_name == "crowding":
                score = -score
            elif signal_name == "valuation_spread":
                score = -score
        known = score.notna()
        ready = ready & known
        scores.append(score)
    frame = pd.concat(scores, axis=1)
    frame.columns = members
    base = float(group["group_weight"]) / len(members)
    strength = (2.0 * tilt) - 1.0
    centered = frame.sub(frame.mean(axis=1), axis=0)
    centered = centered.where(centered.abs().ge(threshold), 0.0)
    raw = base * (1.0 + strength * centered.div(centered.abs().sum(axis=1).replace(0, 1), axis=0))
    cap = group.get("max_active_deviation")
    if cap is not None:
        raw = raw.clip(lower=base - float(cap), upper=base + float(cap))
    raw = raw.clip(lower=0.0)
    total = raw.sum(axis=1).replace(0, pd.NA)
    raw = raw.div(total, axis=0) * float(group["group_weight"])
    return ready, {member: raw[member] for member in members}


def _spread(
    prices: pd.DataFrame,
    pair: dict,
    signals: pd.DataFrame | None,
    window: int,
    signal_name: str,
) -> tuple[pd.Series, pd.Series]:
    if signal_name == "momentum":
        spread = prices[pair["left"]].pct_change(window) - prices[pair["right"]].pct_change(window)
        return spread, spread.notna()
    column = str(pair.get("signal_column") or pair["name"])
    spread = _signal_column(signals, column, prices.index)
    return spread, spread.notna()


def _signal_column(signals: pd.DataFrame | None, column: str, index: pd.Index) -> pd.Series:
    if signals is None or column not in signals.columns:
        raise ValueError(f"style signal column {column} is missing")
    return signals[column].reindex(index)


def _tilt_share(
    spread: pd.Series,
    ready: pd.Series,
    tilt: float,
    threshold: float,
    *,
    favor_left: bool,
) -> pd.Series:
    high = tilt if favor_left else 1.0 - tilt
    low = 1.0 - tilt if favor_left else tilt
    share = pd.Series(0.5, index=spread.index, dtype=float)
    share = share.mask(ready & spread.gt(threshold), high)
    share = share.mask(ready & spread.lt(-threshold), low)
    return share
