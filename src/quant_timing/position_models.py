from __future__ import annotations

import math

import numpy as np
import pandas as pd

from quant_timing.position import apply_external_cap, position_history


def build_position(
    close: pd.Series,
    rules: dict,
    external: pd.Series | None = None,
) -> pd.DataFrame:
    model = str(rules.get("model", "rules"))
    if model == "rules":
        history = position_history(close, rules)
    elif model == "vol_target":
        history = _vol_target(close, rules)
    elif model == "tsmom":
        history = _binary_scale(
            close.pct_change(int(rules["return_window"])),
            rules,
            positive_regime="tsmom",
        )
    elif model == "moving_average":
        fast = close.rolling(int(rules["fast_window"])).mean()
        slow = close.rolling(int(rules["slow_window"])).mean()
        history = _binary_scale(fast - slow, rules, positive_regime="moving_average")
    else:
        raise ValueError(f"unknown position model {model}")
    if external is not None:
        history = apply_external_cap(history, external)
    return history


def _vol_target(close: pd.Series, rules: dict) -> pd.DataFrame:
    returns = close.sort_index().astype(float).pct_change()
    realized = returns.rolling(int(rules["vol_window"])).std() * math.sqrt(252.0)
    target = float(rules["target_vol"])
    floor = float(rules["floor"])
    cap = float(rules["cap"])
    raw = pd.Series(np.nan, index=close.index, dtype=float)
    finite = realized.gt(0)
    raw.loc[finite] = (target / realized.loc[finite]).clip(floor, cap)
    return _frame(close.index, raw, "vol_target", realized_vol=realized)


def _binary_scale(signal: pd.Series, rules: dict, *, positive_regime: str) -> pd.DataFrame:
    floor = float(rules["floor"])
    cap = float(rules["cap"])
    raw = pd.Series(np.nan, index=signal.index, dtype=float)
    known = signal.notna()
    raw.loc[known & signal.gt(0)] = cap
    raw.loc[known & signal.le(0)] = floor
    return _frame(signal.index, raw, positive_regime)


def _frame(
    index: pd.Index,
    scale: pd.Series,
    regime_name: str,
    realized_vol: pd.Series | None = None,
) -> pd.DataFrame:
    warmup = scale.isna()
    regime = pd.Series(regime_name, index=index)
    regime.loc[warmup] = "warmup"
    model_scale = scale.fillna(0.0).astype(float)
    history = pd.DataFrame(
        {
            "regime": regime,
            "model_scale": model_scale,
            "position_scale": model_scale,
            "vol_percentile": None,
            "window_return": None,
            "external_scale": np.nan,
        },
        index=index,
    )
    if realized_vol is not None:
        history["realized_vol"] = realized_vol
    history.index.name = "date"
    return history
