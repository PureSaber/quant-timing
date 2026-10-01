from __future__ import annotations

import pytest

from quant_timing.config import resolve_config
from quant_timing.macro_policy import apply_macro, normalize_macro
from quant_timing.study import run_study
from tests.support import sample_prices, timing_config


def test_complete_macro_without_rules_does_not_change_scale() -> None:
    context = normalize_macro(
        {"complete": True, "unavailable": {}, "values": {"CPI": {"value": "999"}}}
    )
    scale, policy = apply_macro(0.8, context, {"rules": []})
    assert scale == pytest.approx(0.8)
    assert policy == "context_only"


def test_explicit_rule_can_only_cap_scale() -> None:
    context = normalize_macro(
        {"complete": True, "unavailable": {}, "values": {"CPI": {"value": "10"}}}
    )
    rules = [{"series": "CPI", "compare": "above", "level": 1, "scale_cap": 0.4}]
    scale, policy = apply_macro(1.0, context, {"rules": rules})
    assert scale == pytest.approx(0.4)
    assert policy == "rule_cap:CPI"
    unchanged, _ = apply_macro(0.2, context, {"rules": rules})
    assert unchanged == pytest.approx(0.2)


def test_incomplete_macro_requires_a_policy_and_never_zero_fills() -> None:
    context = normalize_macro(
        {"complete": False, "unavailable": {"CPI": "missing_or_stale"}, "values": {}}
    )
    with pytest.raises(ValueError, match="incomplete_policy"):
        apply_macro(1.0, context, {"incomplete_policy": None, "rules": []})
    scale, policy = apply_macro(
        1.0,
        context,
        {"incomplete_policy": "baseline", "baseline_scale": 0.0, "rules": []},
    )
    assert scale == pytest.approx(0.0)
    assert policy == "incomplete_baseline"
    held, held_policy = apply_macro(
        1.0,
        context,
        {"incomplete_policy": "hold_previous", "rules": []},
    )
    assert held is None
    assert held_policy == "incomplete_hold"


def test_contradictory_macro_context_is_rejected() -> None:
    with pytest.raises(ValueError, match="contradicts"):
        normalize_macro({"complete": True, "unavailable": {"CPI": "stale"}, "values": {}})


def test_latest_decision_uses_macro_without_rewriting_history() -> None:
    prices = sample_prices()
    raw = timing_config()
    raw["macro"] = {
        "incomplete_policy": "baseline",
        "baseline_scale": 0.0,
        "rules": [{"series": "CPI", "compare": "above", "level": 1.0, "scale_cap": 0.4}],
    }
    resolve_config(raw)
    plain = run_study(prices, timing_config(), audit=False)
    capped = run_study(
        prices,
        raw,
        macro={"complete": True, "unavailable": {}, "values": {"CPI": {"value": "9"}}},
        audit=False,
    )
    assert capped.position["position_scale"].equals(plain.position["position_scale"])
    assert capped.latest["position_scale"] <= 0.4
    assert capped.latest["signals"]["macro_policy"].startswith("rule_cap")
