from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_timing.contract import ARTIFACT_SCHEMAS, validate_standard_run
from quant_timing.export import exit_code, gate_decision, write_run
from quant_timing.io import load_prices
from quant_timing.study import run_study
from quant_timing.synthetic import write_fixture
from tests.support import sample_prices, timing_config


def test_gate_blocks_a_scale_that_failed_the_causal_audit() -> None:
    latest = {
        "as_of": "2024-01-04",
        "regime": "risk_on",
        "model_position_scale": 1.0,
        "position_scale": 1.0,
        "action": "use",
        "signals": {},
    }
    blocked = gate_decision(latest, {"leakage_passed": False, "scored_folds": 2})
    assert blocked["action"] == "blocked"
    assert blocked["position_scale"] is None
    assert exit_code({"leakage_passed": False, "scored_folds": 2}, blocked) == 1

    short = gate_decision(latest, {"leakage_passed": True, "scored_folds": 0})
    assert short["signals"]["block_reason"] == "insufficient_history"
    assert exit_code({"leakage_passed": True, "scored_folds": 0}, short) == 2


def test_run_writes_a_paper_sim_scale_and_an_immutable_contract(tmp_path) -> None:
    result = run_study(sample_prices(), timing_config(), audit=True)
    decision = write_run(result, sample_prices(), tmp_path / "run")
    assert decision["action"] == "use"
    payload = json.loads((tmp_path / "run" / "position_scale.json").read_text(encoding="utf-8"))
    assert payload["regime"] in {"risk_on", "risk_off", "high_vol"}
    assert 0.0 <= float(payload["position_scale"]) <= 1.0
    overlay = (tmp_path / "run" / "portfolio_overlay.yaml").read_text(encoding="utf-8")
    assert "position_scale" in overlay

    weights = pd.read_csv(tmp_path / "run" / "style_weights.csv")
    value_columns = [column for column in weights.columns if column != "date"]
    assert weights[value_columns].sum(axis=1).sub(1).abs().max() <= 1e-8

    manifest = validate_standard_run(tmp_path / "run")
    assert manifest.project == "quant-timing"
    assert manifest.tags["orders"] == "research_target_not_routed"
    for name, columns in ARTIFACT_SCHEMAS.items():
        header = pd.read_csv(tmp_path / "run" / "standard" / f"{name}.csv", nrows=0).columns
        assert tuple(header[: len(columns)]) == columns
    orders = pd.read_csv(tmp_path / "run" / "standard" / "orders.csv")
    assert not orders.empty
    assert set(orders["status"]) == {"target"}
    summary = (tmp_path / "run" / "validation" / "summary.json").read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_run(result, sample_prices(), tmp_path / "run")
    assert (tmp_path / "run" / "validation" / "summary.json").read_text(encoding="utf-8") == summary


def test_manifest_detects_a_mutated_artifact(tmp_path) -> None:
    result = run_study(sample_prices(), timing_config(), audit=False)
    write_run(result, sample_prices(), tmp_path / "run")
    returns = tmp_path / "run" / "standard" / "returns.csv"
    returns.write_text(returns.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mutated"):
        validate_standard_run(tmp_path / "run")


def test_fixture_round_trip_matches_the_generator(tmp_path) -> None:
    path = tmp_path / "sleeves.csv"
    write_fixture(path)
    loaded = load_prices(path)
    pd.testing.assert_frame_equal(loaded, sample_prices(), check_freq=False, rtol=1e-12, atol=1e-8)
