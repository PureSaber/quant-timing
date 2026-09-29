from __future__ import annotations

import pandas as pd

from quant_timing.macro_policy import apply_macro


def apply_macro_history(
    scale: pd.Series,
    history: pd.DataFrame | None,
    macro_cfg: dict,
) -> pd.Series:
    """Apply explicit macro rules with available_at. Later releases cannot change earlier scales."""
    if history is None or history.empty:
        return scale
    frame = history.copy()
    frame["available_at"] = pd.to_datetime(frame["available_at"])
    frame["value"] = pd.to_numeric(frame["value"], errors="raise")
    rules = list(macro_cfg.get("rules") or [])
    required = [str(rule["series"]) for rule in rules]
    adjusted: list[float] = []
    previous: float | None = None
    for stamp, value in scale.items():
        visible = frame.loc[frame["available_at"] <= pd.Timestamp(stamp)]
        latest = visible.sort_values("available_at").groupby("series", as_index=False).tail(1)
        values = {str(row.series): float(row.value) for row in latest.itertuples(index=False)}
        unavailable = {series: "not_released" for series in required if series not in values}
        context = {
            "complete": not unavailable,
            "values": values,
            "unavailable": unavailable,
        }
        if not rules:
            adjusted.append(float(value))
            previous = float(value)
            continue
        new_scale, _policy = apply_macro(float(value), context, macro_cfg)
        if new_scale is None:
            if previous is None:
                fallback = macro_cfg.get("baseline_scale")
                new_scale = float(value if fallback is None else fallback)
            else:
                new_scale = previous
        adjusted.append(float(new_scale))
        previous = float(new_scale)
    return pd.Series(adjusted, index=scale.index, name=scale.name)
