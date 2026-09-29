from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return payload


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _unit_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return number


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def resolve_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate an explicit timing config. Missing policy is rejected, not defaulted into a signal."""
    if not isinstance(raw, dict):
        raise ValueError("config must be a mapping")
    market = raw.get("market")
    if not isinstance(market, str) or not market or market == "CASH":
        raise ValueError("market must be a non-empty sleeve name other than CASH")

    position_raw = raw.get("position")
    if not isinstance(position_raw, dict):
        raise ValueError("position rules are required")
    position = {
        "vol_window": _positive_int(position_raw.get("vol_window"), "position.vol_window"),
        "vol_lookback": _positive_int(position_raw.get("vol_lookback"), "position.vol_lookback"),
        "vol_percentile_threshold": _unit_number(
            position_raw.get("vol_percentile_threshold"),
            "position.vol_percentile_threshold",
        ),
        "return_window": _positive_int(position_raw.get("return_window"), "position.return_window"),
        "risk_off_return": _finite_number(
            position_raw.get("risk_off_return"), "position.risk_off_return"
        ),
        "high_vol_scale": _unit_number(position_raw.get("high_vol_scale"), "position.high_vol_scale"),
        "risk_off_scale": _unit_number(position_raw.get("risk_off_scale"), "position.risk_off_scale"),
        "risk_on_scale": _unit_number(position_raw.get("risk_on_scale"), "position.risk_on_scale"),
    }

    costs = raw.get("costs")
    if not isinstance(costs, dict):
        raise ValueError("costs are required")
    costs_bps = _finite_number(costs.get("bps"), "costs.bps")
    if costs_bps < 0:
        raise ValueError("costs.bps must be nonnegative")

    validation_raw = raw.get("validation")
    if not isinstance(validation_raw, dict):
        raise ValueError("validation is required")
    validation = {
        "train_size": _positive_int(validation_raw.get("train_size"), "validation.train_size"),
        "test_size": _positive_int(validation_raw.get("test_size"), "validation.test_size"),
        "step_size": _positive_int(validation_raw.get("step_size"), "validation.step_size"),
        "embargo": _nonnegative_int(validation_raw.get("embargo"), "validation.embargo"),
    }

    style = _resolve_style(raw.get("style"))
    sleeves = []
    if style:
        for pair in style["pairs"]:
            sleeves.extend((pair["left"], pair["right"]))
    if market in sleeves:
        raise ValueError("market sleeve cannot also be a style sleeve")

    return {
        "market": market,
        "position": position,
        "costs_bps": costs_bps,
        "validation": validation,
        "style": style,
        "regime": _resolve_regime(raw.get("regime")),
        "macro": _resolve_macro(raw.get("macro")),
        "run_id": _run_id(raw.get("run_id", "timing")),
    }


def _run_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("run_id must be a non-empty string")
    return value


def _resolve_style(raw: object) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("style must be a mapping")
    pairs_raw = raw.get("pairs")
    if not isinstance(pairs_raw, list) or not pairs_raw:
        raise ValueError("style.pairs must be a non-empty list")
    pairs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in pairs_raw:
        if not isinstance(entry, dict):
            raise ValueError("style pair must be a mapping")
        name = entry.get("name")
        left = entry.get("left")
        right = entry.get("right")
        labels = (name, left, right)
        if not all(isinstance(item, str) and item and item != "CASH" for item in labels):
            raise ValueError("style pair names must be non-empty and cannot be CASH")
        if left == right:
            raise ValueError(f"style pair {name} has identical sleeves")
        if left in seen or right in seen:
            raise ValueError(f"sleeve is reused across style pairs: {name}")
        seen.update((left, right))
        pairs.append(
            {
                "name": name,
                "left": left,
                "right": right,
                "group_weight": _unit_number(entry.get("group_weight"), f"style.{name}.group_weight"),
            }
        )
    total = sum(float(pair["group_weight"]) for pair in pairs)
    if not math.isclose(total, 1.0, abs_tol=1e-8):
        raise ValueError("style group weights must sum to 1")
    tilt = _unit_number(raw.get("tilt"), "style.tilt")
    if tilt < 0.5:
        raise ValueError("style.tilt must be between 0.5 and 1")
    threshold = _finite_number(raw.get("threshold"), "style.threshold")
    if threshold < 0:
        raise ValueError("style.threshold must be nonnegative")
    return {
        "return_window": _positive_int(raw.get("return_window"), "style.return_window"),
        "tilt": tilt,
        "threshold": threshold,
        "pairs": pairs,
    }


def _resolve_regime(raw: object) -> dict[str, Any]:
    if raw is None:
        return {"combine": "model", "history": None, "snapshot": None}
    if not isinstance(raw, dict):
        raise ValueError("regime must be a mapping")
    combine = raw.get("combine", "model")
    if combine not in {"model", "cap"}:
        raise ValueError("regime.combine must be model or cap")
    history = raw.get("history")
    snapshot = raw.get("snapshot")
    if history is not None and not isinstance(history, str):
        raise ValueError("regime.history must be a path")
    if snapshot is not None and not isinstance(snapshot, str):
        raise ValueError("regime.snapshot must be a path")
    if history and snapshot:
        raise ValueError("use either regime.history or regime.snapshot, not both")
    if combine == "cap" and not history and not snapshot:
        raise ValueError("regime.combine cap requires history or snapshot")
    if combine == "model" and (history or snapshot):
        raise ValueError("regime path is set but combine is model")
    return {"combine": combine, "history": history, "snapshot": snapshot}


def _resolve_macro(raw: object) -> dict[str, Any]:
    if raw is None:
        return {"context": None, "incomplete_policy": None, "baseline_scale": None, "rules": []}
    if not isinstance(raw, dict):
        raise ValueError("macro must be a mapping")
    context = raw.get("context")
    if context is not None and not isinstance(context, str):
        raise ValueError("macro.context must be a path")
    policy = raw.get("incomplete_policy")
    if policy is not None and policy not in {"baseline", "hold_previous"}:
        raise ValueError("macro.incomplete_policy must be baseline or hold_previous")
    baseline = raw.get("baseline_scale")
    baseline_scale = None if baseline is None else _unit_number(baseline, "macro.baseline_scale")
    if policy == "baseline" and baseline_scale is None:
        raise ValueError("macro.baseline_scale is required when policy is baseline")
    rules = []
    for entry in raw.get("rules") or []:
        if not isinstance(entry, dict):
            raise ValueError("macro rule must be a mapping")
        series = entry.get("series")
        compare = entry.get("compare")
        if not isinstance(series, str) or not series:
            raise ValueError("macro rule series is required")
        if compare not in {"above", "below"}:
            raise ValueError("macro rule compare must be above or below")
        rules.append(
            {
                "series": series,
                "compare": compare,
                "level": _finite_number(entry.get("level"), f"macro.{series}.level"),
                "scale_cap": _unit_number(entry.get("scale_cap"), f"macro.{series}.scale_cap"),
            }
        )
    return {
        "context": context,
        "incomplete_policy": policy,
        "baseline_scale": baseline_scale,
        "rules": rules,
    }
