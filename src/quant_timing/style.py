from __future__ import annotations

import pandas as pd


def style_sleeves(style: dict) -> list[str]:
    names: list[str] = []
    for pair in style["pairs"]:
        names.extend((pair["left"], pair["right"]))
    return names


def style_internal_weights(prices: pd.DataFrame, style: dict) -> pd.DataFrame:
    """Long-only pair weights known at the close. Each row sums to 1 once every pair is ready."""
    window = int(style["return_window"])
    tilt = float(style["tilt"])
    threshold = float(style["threshold"])
    sleeves = style_sleeves(style)
    weights = pd.DataFrame(index=prices.index, columns=sleeves, dtype=float)
    ready = pd.Series(True, index=prices.index)
    for pair in style["pairs"]:
        left_return = prices[pair["left"]].pct_change(window)
        right_return = prices[pair["right"]].pct_change(window)
        spread = left_return - right_return
        pair_ready = spread.notna()
        ready = ready & pair_ready
        left_share = pd.Series(0.5, index=prices.index, dtype=float)
        left_share = left_share.mask(pair_ready & spread.gt(threshold), tilt)
        left_share = left_share.mask(pair_ready & spread.lt(-threshold), 1.0 - tilt)
        group_weight = float(pair["group_weight"])
        left_weight = group_weight * left_share
        weights[pair["left"]] = left_weight
        weights[pair["right"]] = group_weight - left_weight
    weights.loc[~ready, :] = pd.NA
    weights = weights.astype(float)
    weights.index.name = "date"
    return weights
