from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _finite(value: object) -> float:
    try:
        if value is None or pd.isna(value):
            return float("nan")
    except TypeError:
        return float("nan")
    number = float(value)
    if not math.isfinite(number):
        return float("nan")
    return number


def classify_position(vol_percentile: object, window_return: object, rules: dict) -> tuple[str, float]:
    """Map features known on the decision date to a long-only scale.

    High volatility takes priority over a negative window return. Incomplete features stay in
    cash. They are not filled forward into the previous risk-on label.
    """
    vol_value = _finite(vol_percentile)
    return_value = _finite(window_return)
    if math.isnan(vol_value) or math.isnan(return_value):
        return "warmup", 0.0
    if vol_value >= float(rules["vol_percentile_threshold"]):
        return "high_vol", float(rules["high_vol_scale"])
    if return_value < float(rules["risk_off_return"]):
        return "risk_off", float(rules["risk_off_scale"])
    return "risk_on", float(rules["risk_on_scale"])


def _percent_rank(values: np.ndarray) -> float:
    if np.isnan(values).any():
        return float("nan")
    return float(pd.Series(values).rank(pct=True).iloc[-1])


def position_history(
    close: pd.Series,
    rules: dict,
    external: pd.Series | None = None,
) -> pd.DataFrame:
    if close.empty:
        raise ValueError("market close series is empty")
    close = close.sort_index().astype(float)
    returns = close.pct_change()
    vol = returns.rolling(int(rules["vol_window"])).std() * math.sqrt(252.0)
    vol_pct = vol.rolling(int(rules["vol_lookback"])).apply(_percent_rank, raw=True)
    window_return = close.pct_change(int(rules["return_window"]))
    rows = []
    for stamp, vol_value, return_value in zip(
        close.index, vol_pct.to_numpy(), window_return.to_numpy(), strict=True
    ):
        regime, scale = classify_position(vol_value, return_value, rules)
        rows.append(
            {
                "regime": regime,
                "model_scale": scale,
                "position_scale": scale,
                "vol_percentile": None if _missing(vol_value) else float(vol_value),
                "window_return": None if _missing(return_value) else float(return_value),
                "external_scale": np.nan,
            }
        )
    history = pd.DataFrame(rows, index=close.index)
    history.index.name = "date"
    if external is not None:
        history = apply_external_cap(history, external)
    return history


def _missing(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def apply_external_cap(history: pd.DataFrame, external: pd.Series) -> pd.DataFrame:
    """Cap model scale by the last published external scale at or before each date.

    A scale published after the decision date is not eligible. Mature decisions with no
    eligible external scale fail instead of falling back to the uncapped model.
    """
    if external.empty:
        raise ValueError("external position_scale history is empty")
    aligned = external.sort_index().reindex(history.index, method="ffill")
    mature = history["regime"].ne("warmup")
    if (mature & aligned.isna()).any():
        raise ValueError("external position_scale does not cover every mature decision")
    capped = history.copy()
    capped["external_scale"] = aligned
    capped.loc[mature, "position_scale"] = np.minimum(
        capped.loc[mature, "model_scale"].to_numpy(dtype=float),
        aligned.loc[mature].to_numpy(dtype=float),
    )
    return capped
