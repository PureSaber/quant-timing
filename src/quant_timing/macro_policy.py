from __future__ import annotations

import math
from typing import Any


def normalize_macro(context: dict | None) -> dict | None:
    if context is None:
        return None
    if not isinstance(context, dict):
        raise ValueError("macro context must be a mapping")
    if "complete" not in context or "values" not in context:
        raise ValueError("macro context requires complete and values")
    complete = context["complete"]
    if not isinstance(complete, bool):
        raise ValueError("macro complete must be a boolean")
    values = context["values"]
    if not isinstance(values, dict):
        raise ValueError("macro values must be a mapping")
    unavailable = context.get("unavailable", {})
    if not isinstance(unavailable, dict):
        raise ValueError("macro unavailable must be a mapping")
    if complete and unavailable:
        raise ValueError("macro context contradicts itself")
    return {"complete": complete, "values": values, "unavailable": unavailable}


def apply_macro(scale: float, context: dict | None, macro_cfg: dict) -> tuple[float | None, str]:
    """Apply an explicit macro policy. A complete context without rules does not change scale."""
    if context is None:
        return scale, "no_macro"
    if not context["complete"]:
        policy = macro_cfg.get("incomplete_policy")
        if policy == "baseline":
            return float(macro_cfg["baseline_scale"]), "incomplete_baseline"
        if policy == "hold_previous":
            return None, "incomplete_hold"
        raise ValueError("incomplete macro requires an explicit incomplete_policy")
    capped = scale
    applied: list[str] = []
    for rule in macro_cfg.get("rules") or []:
        number = _series_value(context["values"], str(rule["series"]))
        level = float(rule["level"])
        if rule["compare"] == "above":
            triggered = number > level
        else:
            triggered = number < level
        if triggered:
            capped = min(capped, float(rule["scale_cap"]))
            applied.append(str(rule["series"]))
    if applied:
        return capped, "rule_cap:" + ",".join(applied)
    return scale, "context_only"


def _series_value(values: dict[str, Any], series: str) -> float:
    if series not in values:
        raise ValueError(f"complete macro context is missing {series}")
    raw = values[series]
    if isinstance(raw, dict):
        if "value" not in raw:
            raise ValueError(f"macro series {series} has no value")
        raw = raw["value"]
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        raise ValueError(f"macro series {series} is not numeric")
    try:
        number = float(raw)
    except ValueError as exc:
        raise ValueError(f"macro series {series} is not numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"macro series {series} is not finite")
    return number
