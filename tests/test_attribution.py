import json
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_timing.attribution import write_attribution
from quant_timing.book import simulate
from quant_timing.contract import file_sha256, validate_standard_run
from quant_timing.export import _standard_frames, write_run
from quant_timing.study import run_study
from quant_timing.walkforward import iter_folds, score_folds


def _manual(weights, matched, prices, *, cash=None, extras=None, bps=100, impact=0.2):
    realized = simulate(weights, matched, prices, "A", bps, cash, impact, extras)
    folds = score_folds(
        realized, iter_folds(prices.index, train_size=1, test_size=2, step_size=1, embargo=0)
    )
    return SimpleNamespace(weights=weights, matched_weights=matched, realized=realized, folds=folds)


def test_linked_asset_cash_and_cost_contributions_match_hand_calculation(tmp_path):
    dates = pd.bdate_range("2026-01-01", periods=3)
    prices = pd.DataFrame({"A": [100, 110, 99]}, index=dates)
    weights = pd.DataFrame({"A": 0.5, "CASH": 0.5}, index=dates)
    matched = pd.DataFrame({"A": 0.75, "CASH": 0.25}, index=dates)
    result = _manual(weights, matched, prices, cash=pd.Series([0, 0.02, 0.01], index=dates))
    frames = _standard_frames(result, prices)
    payload = write_attribution(result, tmp_path, frames["returns"])
    full = payload["periods"][0]
    components = {row["component"]: row for row in full["components"]}
    turnover = 0.55 / 1.06 - 0.5
    base_cost = turnover * 0.01
    impact_cost = 0.2 * turnover**1.5
    assert components["A"]["strategy"] == pytest.approx(0.05 - 1.06 * 0.05)
    assert components["CASH"]["strategy"] == pytest.approx(0.01 + 1.06 * 0.005)
    assert components["base_cost"]["strategy"] == pytest.approx(-1.06 * base_cost)
    assert components["impact_cost"]["strategy"] == pytest.approx(-1.06 * impact_cost)
    expected = 1.06 * (1 - 0.045 - base_cost - impact_cost) - 1
    assert full["net_return"] == pytest.approx(expected)
    assert sum(row["strategy"] for row in full["components"]) == pytest.approx(expected)
    assert sum(row["difference"] for row in full["components"]) == pytest.approx(
        full["matched_excess_return"]
    )
    # The last decision has no following return, and the fold starts from its own unit NAV.
    fold = payload["periods"][1]
    assert fold["n_decisions"] == 1
    assert fold["first_return_date"] == str(dates[-1].date())
    assert fold["net_return"] == pytest.approx(-0.045 - base_cost - impact_cost)
    assert frames["costs"].iloc[-1]["market_impact"] == pytest.approx(1.06 * impact_cost)
    assert frames["costs"].iloc[-1]["slippage"] == pytest.approx(1.06 * base_cost)


def test_futures_pnl_cash_and_margin_are_separate_without_funded_notional(tmp_path):
    dates = pd.bdate_range("2026-01-01", periods=3)
    prices = pd.DataFrame({"A": 100.0, "IF": [100, 110, 110]}, index=dates)
    weights = pd.DataFrame({"A": 0.9, "IF": -0.5, "CASH": 0.05, "MARGIN": 0.05}, index=dates)
    result = _manual(
        weights,
        weights,
        prices,
        bps=0,
        impact=0,
        cash=pd.Series([0, 0.02, 0], index=dates),
        extras={"IF": prices["IF"].pct_change()},
    )
    full = write_attribution(result, tmp_path, _standard_frames(result, prices)["returns"])[
        "periods"
    ][0]
    components = {row["component"]: row for row in full["components"]}
    assert components["IF"]["strategy"] == pytest.approx(-0.05)
    assert components["IF"]["kind"] == "futures"
    assert components["CASH"]["strategy"] == pytest.approx(0.001)
    assert components["MARGIN"]["strategy"] == 0
    assert full["net_return"] == pytest.approx(-0.049)


def _export(tmp_path, *, overlap=False):
    dates = pd.bdate_range("2026-01-01", periods=12)
    prices = pd.DataFrame(
        {"A": [100, 110, 90, 99, 108, 100, 95, 105, 100, 102, 99, 103]}, index=dates
    )
    raw = {
        "market": "A",
        "position": {"model": "tsmom", "return_window": 1, "floor": 0.5, "cap": 0.5},
        "costs": {"bps": 20, "impact_coef": 0.1},
        "validation": {
            "train_size": 2,
            "test_size": 3,
            "step_size": 1 if overlap else 3,
            "embargo": 1,
        },
    }
    result = run_study(prices, raw)
    out = tmp_path / "run"
    write_run(result, prices, out)
    return out, result


def _refresh_metrics(out, metrics):
    path = out / "standard/metrics.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    manifest_path = out / "standard/run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["artifacts"]:
        if record["name"] == "metrics":
            record["sha256"] = file_sha256(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_overlapping_folds_are_individually_rebased_not_aggregated(tmp_path):
    out, result = _export(tmp_path, overlap=True)
    validate_standard_run(out)
    metrics = json.loads((out / "standard/metrics.json").read_text(encoding="utf-8"))
    attribution = metrics["return_attribution"]
    assert attribution["overlapping_folds_are_not_aggregated"] is True
    assert attribution["folds_are_separately_executed_accounts"] is False
    assert len(attribution["periods"]) == len(result.folds) + 1
    for period, fold in zip(attribution["periods"][1:], result.folds.to_dict(orient="records")):
        assert period["net_return"] == pytest.approx(fold["net_return"])
        assert sum(row["strategy"] for row in period["components"]) == pytest.approx(
            fold["net_return"]
        )
        assert period["first_return_date"] > fold["test_start"]


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "mutated",
        "weight",
        "pnl",
        "decision",
        "duplicate",
        "cost",
        "summary",
        "semantics",
    ],
)
def test_missing_corrupt_and_internally_inconsistent_attribution_is_rejected(tmp_path, change):
    out, _ = _export(tmp_path)
    path = out / "attribution/daily.csv"
    metrics = json.loads((out / "standard/metrics.json").read_text(encoding="utf-8"))
    if change == "missing":
        path.unlink()
    elif change == "mutated":
        path.write_bytes(path.read_bytes() + b"\n")
    elif change == "summary":
        metrics["return_attribution"]["periods"][0]["components"][0]["strategy"] += 0.01
        _refresh_metrics(out, metrics)
    elif change == "semantics":
        metrics["return_attribution"]["cost_semantics"]["observed_execution_slippage"] = True
        _refresh_metrics(out, metrics)
    else:
        frame = pd.read_csv(path)
        if change == "weight":
            frame.loc[0, "return_weight"] += 0.1
        elif change == "pnl":
            frame.loc[0, "pnl"] += 0.1
        elif change == "decision":
            frame.loc[0, "decision_date"] = frame.loc[0, "date"]
        elif change == "duplicate":
            frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        elif change == "cost":
            loc = frame.index[frame["kind"] == "modeled_cost"][-1]
            frame.loc[loc, "contribution"] = 0.1
            frame.loc[loc, "pnl"] = 0.1 * frame.loc[loc, "opening_nav"]
        frame.to_csv(path, index=False)
        metrics["return_attribution"]["daily"]["sha256"] = file_sha256(path)
        metrics["return_attribution"]["daily"]["rows"] = len(frame)
        _refresh_metrics(out, metrics)
    with pytest.raises(ValueError, match="attribution"):
        validate_standard_run(out)


def test_export_rejects_unreconciled_native_components_without_publishing(tmp_path):
    out, result = _export(tmp_path)
    result.realized.loc[result.realized.index[2], "contribution:A"] += 0.01
    with pytest.raises(ValueError, match="attribution"):
        write_run(result, result.valuation_prices, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    validate_standard_run(out)


def test_existing_runs_without_attribution_remain_readable_but_new_binding_is_required(tmp_path):
    out, _ = _export(tmp_path)
    metrics = json.loads((out / "standard/metrics.json").read_text(encoding="utf-8"))
    del metrics["return_attribution"]
    _refresh_metrics(out, metrics)
    with pytest.raises(ValueError, match="binding"):
        validate_standard_run(out)
    path = out / "standard/run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["tags"]["return_attribution"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    validate_standard_run(out)
