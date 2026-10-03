from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from quant_timing.cli import main
from quant_timing.contract import file_sha256
from quant_timing.counterfactual import (
    intervention_configs,
    run_counterfactual,
    validate_counterfactual,
)
from quant_timing.study import run_study
from tests.support import timing_config


def _case(root, *, style=False, overlay=False):
    dates = pd.bdate_range("2026-01-01", periods=13)
    prices = pd.DataFrame(
        {
            "market": 100 * np.cumprod([1, *([1.1, 0.9] * 6)]),
            "left": np.linspace(100, 130, 13),
            "right": np.linspace(100, 90, 13),
        },
        index=dates.rename("date"),
    )
    prices.to_csv(root / "prices.csv")
    raw = {
        "market": "market",
        "run_id": "paired-test",
        "source": {"evidence_kind": "synthetic"},
        "input": {"path": "prices.csv", "format": "wide", "date_col": "date"},
        "position": {"model": "tsmom", "return_window": 1, "floor": 0, "cap": 1},
        "costs": {"bps": 0, "impact_coef": 0},
        "validation": {"train_size": 2, "test_size": 4, "step_size": 2, "embargo": 1},
    }
    if style:
        raw["style"] = {
            "return_window": 1,
            "tilt": 0.8,
            "threshold": 0,
            "pairs": [{"name": "pair", "left": "left", "right": "right", "group_weight": 1}],
        }
    if overlay:
        pd.DataFrame({"date": dates, "IF": np.linspace(100, 110, 13)}).to_csv(
            root / "futures.csv", index=False
        )
        raw["futures"] = {"path": "futures.csv"}
        raw["overlay"] = {
            "mode": "futures",
            "contracts": ["IF"],
            "margin_rate": 0.1,
            "beta_window": 3,
        }
    config = root / "config.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return config, raw, prices


@pytest.mark.parametrize("model", ["rules", "vol_target", "tsmom", "moving_average"])
def test_interventions_change_only_declared_model_fields_and_identity(model):
    raw = timing_config()
    raw["constraints"] = {"max_daily_change": 0.1, "drawdown_line": 0.15, "drawdown_floor": 0.2}
    if model != "rules":
        raw["position"] = {
            "model": model,
            "floor": 0.1,
            "cap": 0.8,
            "vol_window": 5,
            "target_vol": 0.15,
            "return_window": 5,
            "fast_window": 2,
            "slow_window": 5,
        }
    before = deepcopy(raw)
    variants = intervention_configs(raw)
    assert raw == before and variants["base"] == raw
    for name, candidate in variants.items():
        expected = deepcopy(raw)
        if name in {"position_neutral", "joint"}:
            keys = (
                ("high_vol_scale", "risk_off_scale", "risk_on_scale")
                if model == "rules"
                else ("floor", "cap")
            )
            for key in keys:
                expected["position"][key] = 1
        if name in {"style_neutral", "joint"}:
            expected["style"]["tilt"] = 0.5
        if name != "base":
            expected["run_id"] = f"test--{name}"
        assert candidate == expected


def test_hand_calculated_effect_and_original_warmup_and_constraints(tmp_path):
    _, raw, prices = _case(tmp_path)
    prices = prices.iloc[:7]
    variants = intervention_configs(raw)
    baseline = run_study(prices, variants["base"])
    neutral = run_study(prices, variants["position_neutral"])
    assert baseline.summary["descriptive_full_sample_net"] == pytest.approx(0.9**3 - 1)
    assert neutral.summary["descriptive_full_sample_net"] == pytest.approx(0.9**3 * 1.1**2 - 1)
    assert neutral.position.iloc[0]["regime"] == "warmup"
    assert neutral.weights.iloc[0]["CASH"] == 1
    raw["regime"] = {"combine": "cap", "history": "external.csv"}
    raw["constraints"] = {"max_daily_change": 0.1}
    constrained = run_study(
        prices,
        intervention_configs(raw)["position_neutral"],
        regime_history=pd.Series(0.4, index=prices.index),
    )
    assert constrained.position["model_scale"].iloc[1:].eq(1).all()
    assert constrained.position["position_scale"].max() <= 0.4
    assert constrained.position["position_scale"].diff().abs().max() <= 0.1 + 1e-12


@pytest.mark.parametrize("style,overlay", [(False, False), (True, False), (False, True)])
def test_family_exports_native_ledgers_reconciles_and_never_publishes(tmp_path, style, overlay):
    config, _, _ = _case(tmp_path, style=style, overlay=overlay)
    if overlay:
        raw = yaml.safe_load(config.read_text())
        raw["costs"] = {"bps": 10, "impact_coef": 0.1}
        config.write_text(yaml.safe_dump(raw))
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in tmp_path.iterdir()}
    out = tmp_path / "out"
    result = run_counterfactual(config, out)
    assert result["status"] == "complete"
    assert result["evidence_kind"] == "synthetic"
    assert result["style_applicable"] is style
    assert len(result["candidates"]) == (4 if style else 2)
    assert len(result["periods"]) > 2
    for period in result["periods"]:
        effects = period["effects"]
        for effect in effects.values():
            assert sum(effect["component_effects"].values()) == pytest.approx(effect["effect"])
            assert abs(effect["reconciliation_residual"]) < 1e-10
        if style:
            assert period["interaction"] == pytest.approx(
                effects["joint"]["net_return"]
                - effects["position_neutral"]["net_return"]
                - effects["style_neutral"]["net_return"]
                + effects["base"]["net_return"]
            )
        else:
            assert period["interaction"] is None
    assert not list(out.rglob("position_scale.json"))
    assert not list(out.rglob("portfolio_overlay.yaml"))
    assert all(
        (path.read_bytes(), path.stat().st_mtime_ns) == value for path, value in before.items()
    )
    assert validate_counterfactual(out) == result
    assert main(["verify-counterfactual", "--run-dir", str(out)]) == 0
    with pytest.raises(FileExistsError):
        run_counterfactual(config, out)


def test_protocol_precedes_all_runs_and_one_failure_blocks_family(tmp_path, monkeypatch):
    import quant_timing.counterfactual as paired

    config, _, _ = _case(tmp_path, style=True)
    native = paired.run_study
    seen = []

    def fail_one(prices, raw, **extras):
        protocol = json.loads((tmp_path / "out/protocol.json").read_text(encoding="utf-8"))
        assert len(protocol["variants"]) == 4
        seen.append(raw["run_id"])
        if raw["run_id"].endswith("--style_neutral"):
            raise ValueError("injected candidate failure")
        return native(prices, raw, **extras)

    monkeypatch.setattr(paired, "run_study", fail_one)
    result = run_counterfactual(config, tmp_path / "out")
    assert len(seen) == 4
    assert result["status"] == "incomplete" and result["periods"] == []
    assert result["candidates"]["style_neutral"]["status"] == "failed"
    assert "injected candidate failure" in (tmp_path / "out/report.html").read_text(
        encoding="utf-8"
    )
    assert main(["verify-counterfactual", "--run-dir", str(tmp_path / "out")]) == 2


@pytest.mark.parametrize(
    "mutation", ["missing", "manifest", "summary", "protocol", "report", "publish"]
)
def test_mutated_or_missing_evidence_is_rejected(tmp_path, mutation):
    config, _, _ = _case(tmp_path)
    out = tmp_path / "out"
    run_counterfactual(config, out)
    if mutation == "missing":
        (out / "runs/position_neutral/attribution/daily.csv").unlink()
    elif mutation == "manifest":
        path = out / "runs/position_neutral/standard/run_manifest.json"
        path.write_text(path.read_text() + " ")
    elif mutation in {"summary", "protocol"}:
        path = out / ("result.json" if mutation == "summary" else "protocol.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if mutation == "summary":
            payload["periods"][0]["effects"]["position_neutral"]["effect"] += 1
        else:
            payload["variants"]["position_neutral"]["costs"]["bps"] = 1
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "report":
        (out / "report.html").write_text("false success")
    else:
        (out / "runs/base/position_scale.json").write_text('{"position_scale":1}')
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_counterfactual(out)


def test_rehashed_out_of_scope_protocol_is_rejected(tmp_path):
    config, _, _ = _case(tmp_path)
    out = tmp_path / "out"
    run_counterfactual(config, out)
    path = out / "protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    protocol["variants"]["position_neutral"]["costs"]["bps"] = 0.5
    path.write_text(json.dumps(protocol), encoding="utf-8")
    receipt_path = out / "result.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["protocol_sha256"] = file_sha256(path)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed intervention family"):
        validate_counterfactual(out)


def test_changed_source_blocks_remaining_candidates_and_retains_failure(tmp_path, monkeypatch):
    import quant_timing.counterfactual as paired

    config, _, _ = _case(tmp_path, style=True)
    original = paired.run_study
    calls = []

    def mutate_after_execution(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(1)
        with (tmp_path / "prices.csv").open("a") as source:
            source.write("\n")
        return result

    monkeypatch.setattr(paired, "run_study", mutate_after_execution)
    result = paired.run_counterfactual(config, tmp_path / "out")
    assert len(calls) == 1
    assert result["status"] == "incomplete" and result["periods"] == []
    assert all(record["status"] == "failed" for record in result["candidates"].values())
    assert "sources changed" in result["source_error"]
    assert not list((tmp_path / "out").rglob("position_scale.json"))


def test_cli_runs_and_reports_malformed_receipt_without_success(tmp_path):
    config, _, _ = _case(tmp_path)
    out = tmp_path / "out"
    assert main(["counterfactual", "--config", str(config), "--out", str(out)]) == 0
    receipt_path = out / "result.json"
    receipt_path.write_text("{}")
    assert main(["verify-counterfactual", "--run-dir", str(out)]) == 2
