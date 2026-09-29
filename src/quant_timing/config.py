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

    position = _resolve_position(raw.get("position"))

    costs = raw.get("costs")
    if not isinstance(costs, dict):
        raise ValueError("costs are required")
    costs_bps = _finite_number(costs.get("bps"), "costs.bps")
    if costs_bps < 0:
        raise ValueError("costs.bps must be nonnegative")
    impact_coef = _finite_number(costs.get("impact_coef", 0.0), "costs.impact_coef")
    if impact_coef < 0:
        raise ValueError("costs.impact_coef must be nonnegative")

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
        "impact_coef": impact_coef,
        "validation": validation,
        "style": style,
        "regime": _resolve_regime(raw.get("regime")),
        "macro": _resolve_macro(raw.get("macro")),
        "constraints": _resolve_constraints(raw.get("constraints")),
        "overlay": _resolve_overlay(raw.get("overlay")),
        "cash": _resolve_cash(raw.get("cash")),
        "run_id": _run_id(raw.get("run_id", "timing")),
    }


def _resolve_position(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("position rules are required")
    model = raw.get("model", "rules")
    if model not in {"rules", "vol_target", "tsmom", "moving_average"}:
        raise ValueError("position.model must be rules, vol_target, tsmom, or moving_average")
    if model == "rules":
        return {
            "model": model,
            "vol_window": _positive_int(raw.get("vol_window"), "position.vol_window"),
            "vol_lookback": _positive_int(raw.get("vol_lookback"), "position.vol_lookback"),
            "vol_percentile_threshold": _unit_number(
                raw.get("vol_percentile_threshold"),
                "position.vol_percentile_threshold",
            ),
            "return_window": _positive_int(raw.get("return_window"), "position.return_window"),
            "risk_off_return": _finite_number(raw.get("risk_off_return"), "position.risk_off_return"),
            "high_vol_scale": _unit_number(raw.get("high_vol_scale"), "position.high_vol_scale"),
            "risk_off_scale": _unit_number(raw.get("risk_off_scale"), "position.risk_off_scale"),
            "risk_on_scale": _unit_number(raw.get("risk_on_scale"), "position.risk_on_scale"),
        }
    floor = _unit_number(raw.get("floor", 0.0), "position.floor")
    cap = _unit_number(raw.get("cap", 1.0), "position.cap")
    if floor > cap:
        raise ValueError("position.floor cannot exceed position.cap")
    resolved = {"model": model, "floor": floor, "cap": cap}
    if model == "vol_target":
        target = _finite_number(raw.get("target_vol"), "position.target_vol")
        if target <= 0:
            raise ValueError("position.target_vol must be positive")
        resolved["target_vol"] = target
        resolved["vol_window"] = _positive_int(raw.get("vol_window"), "position.vol_window")
    elif model == "tsmom":
        resolved["return_window"] = _positive_int(raw.get("return_window"), "position.return_window")
    else:
        fast = _positive_int(raw.get("fast_window"), "position.fast_window")
        slow = _positive_int(raw.get("slow_window"), "position.slow_window")
        if fast >= slow:
            raise ValueError("position.fast_window must be shorter than position.slow_window")
        resolved["fast_window"] = fast
        resolved["slow_window"] = slow
    return resolved


def _run_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("run_id must be a non-empty string")
    return value


def _resolve_style(raw: object) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("style must be a mapping")
    pairs = [_resolve_pair(entry) for entry in raw.get("pairs") or []]
    groups = [_resolve_group(entry) for entry in raw.get("groups") or []]
    if not pairs and not groups:
        raise ValueError("style needs pairs or groups")
    seen: set[str] = set()
    for pair in pairs:
        for sleeve in (pair["left"], pair["right"]):
            if sleeve in seen:
                raise ValueError(f"sleeve is reused: {sleeve}")
            seen.add(sleeve)
    for group in groups:
        for sleeve in group["members"]:
            if sleeve in seen:
                raise ValueError(f"sleeve is reused: {sleeve}")
            seen.add(sleeve)
    total = sum(float(item["group_weight"]) for item in [*pairs, *groups])
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
        "groups": groups,
    }


def _resolve_pair(entry: object) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError("style pair must be a mapping")
    name = entry.get("name")
    left = entry.get("left")
    right = entry.get("right")
    labels = (name, left, right)
    if not all(isinstance(item, str) and item and item not in {"CASH", "FUTURES", "MARGIN"} for item in labels):
        raise ValueError("style pair names must be non-empty sleeves")
    if left == right:
        raise ValueError(f"style pair {name} has identical sleeves")
    signal = entry.get("signal", "momentum")
    if signal not in {"momentum", "valuation_spread", "revision", "crowding"}:
        raise ValueError(f"style pair {name} has an unknown signal")
    cap = entry.get("max_active_deviation")
    return {
        "name": name,
        "left": left,
        "right": right,
        "group_weight": _unit_number(entry.get("group_weight"), f"style.{name}.group_weight"),
        "signal": signal,
        "signal_column": entry.get("signal_column") or name,
        "max_active_deviation": None
        if cap is None
        else _unit_number(cap, f"style.{name}.max_active_deviation"),
    }


def _resolve_group(entry: object) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError("style group must be a mapping")
    name = entry.get("name")
    members = entry.get("members")
    if not isinstance(name, str) or not name:
        raise ValueError("style group name is required")
    if not isinstance(members, list) or len(members) < 2:
        raise ValueError(f"style group {name} needs at least two members")
    if any(not isinstance(member, str) or member in {"CASH", "FUTURES", "MARGIN"} for member in members):
        raise ValueError(f"style group {name} has an invalid member")
    if len(set(members)) != len(members):
        raise ValueError(f"style group {name} repeats a member")
    signal = entry.get("signal", "momentum")
    if signal not in {"momentum", "valuation_spread", "revision", "crowding"}:
        raise ValueError(f"style group {name} has an unknown signal")
    cap = entry.get("max_active_deviation")
    return {
        "name": name,
        "members": list(members),
        "group_weight": _unit_number(entry.get("group_weight"), f"style.{name}.group_weight"),
        "signal": signal,
        "max_active_deviation": None
        if cap is None
        else _unit_number(cap, f"style.{name}.max_active_deviation"),
    }


def _resolve_constraints(raw: object) -> dict[str, Any]:
    if raw is None:
        return {"max_daily_change": None, "drawdown_line": None, "drawdown_floor": 0.0}
    if not isinstance(raw, dict):
        raise ValueError("constraints must be a mapping")
    step = raw.get("max_daily_change")
    line = raw.get("drawdown_line")
    floor = _unit_number(raw.get("drawdown_floor", 0.0), "constraints.drawdown_floor")
    return {
        "max_daily_change": None
        if step is None
        else _unit_number(step, "constraints.max_daily_change"),
        "drawdown_line": None if line is None else _unit_number(line, "constraints.drawdown_line"),
        "drawdown_floor": floor,
    }


def _resolve_overlay(raw: object) -> dict[str, Any]:
    if raw is None:
        return {"mode": "cash", "price": None, "margin_rate": 0.0}
    if not isinstance(raw, dict):
        raise ValueError("overlay must be a mapping")
    mode = raw.get("mode", "cash")
    if mode not in {"cash", "futures"}:
        raise ValueError("overlay.mode must be cash or futures")
    price = raw.get("price")
    if mode == "futures" and not isinstance(price, str):
        raise ValueError("overlay.price is required for futures")
    margin = _unit_number(raw.get("margin_rate", 0.0), "overlay.margin_rate")
    if margin >= 1:
        raise ValueError("overlay.margin_rate must be below 1")
    return {"mode": mode, "price": price, "margin_rate": margin}


def _resolve_cash(raw: object) -> dict[str, Any]:
    if raw is None:
        return {"mode": "zero", "yield_column": None, "daycount": 252}
    if not isinstance(raw, dict):
        raise ValueError("cash must be a mapping")
    mode = raw.get("mode", "zero")
    if mode not in {"zero", "yield", "price"}:
        raise ValueError("cash.mode must be zero, yield, or price")
    column = raw.get("yield_column")
    if mode in {"yield", "price"} and not isinstance(column, str):
        raise ValueError("cash.yield_column is required")
    return {
        "mode": mode,
        "yield_column": column,
        "daycount": _positive_int(raw.get("daycount", 252), "cash.daycount"),
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
        return {"context": None, "history": None, "incomplete_policy": None, "baseline_scale": None, "rules": []}
    if not isinstance(raw, dict):
        raise ValueError("macro must be a mapping")
    context = raw.get("context")
    history = raw.get("history")
    if context is not None and not isinstance(context, str):
        raise ValueError("macro.context must be a path")
    if history is not None and not isinstance(history, str):
        raise ValueError("macro.history must be a path")
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
        "history": history,
        "incomplete_policy": policy,
        "baseline_scale": baseline_scale,
        "rules": rules,
    }
