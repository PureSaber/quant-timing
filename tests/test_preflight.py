from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml

import quant_timing.cli as cli
import quant_timing.preflight as preflight
import quant_timing.study as study
from quant_timing.contract import validate_standard_run
from quant_timing.export import write_run
from quant_timing.preflight import identify_sources, prepare_inputs
from tests.support import sample_prices, timing_config


@pytest.fixture
def config_file(tmp_path):
    prices = tmp_path / "prices.csv"
    sample_prices().to_csv(prices, index_label="date")
    config = timing_config()
    config["input"] = {"path": "prices.csv"}
    config["source"] = {"evidence_kind": "synthetic"}
    path = tmp_path / "study.yaml"
    save_config(path, config)
    return path


def save_config(path, config):
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_preflight_is_read_only_and_does_not_compute(config_file, monkeypatch, capsys):
    before = identify_sources({p.name: p for p in config_file.parent.iterdir()})

    def forbidden(*args, **kwargs):
        pytest.fail("preflight must not run models, simulate, audit or export")

    for module, names in (
        (cli, ("run_study", "write_run")),
        (study, ("build_position", "style_internal_weights", "simulate", "_audit")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    assert cli.main(["preflight", "--config", str(config_file)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["software_preflight"] == "pass"
    assert report["read_only"] and not report["investable"]
    assert report["evidence_kind"] == "synthetic"
    assert report["rows"] == len(sample_prices())
    assert report["folds"] > 0
    assert report["performance_basis"] == "descriptive_full_sample_not_walk_forward_acceptance"
    after = identify_sources({p.name: p for p in config_file.parent.iterdir()})
    assert after == before


def test_native_run_matches_existing_research_api(config_file, tmp_path):
    prepared = prepare_inputs(config_file)
    expected = study.run_study(prepared.prices, prepared.raw, **prepared.extras)
    write_run(expected, prepared.prices, tmp_path / "expected")
    actual = tmp_path / "actual"
    assert cli.main(["run", "--config", str(config_file), "--out", str(actual)]) == 0
    validate_standard_run(actual)
    for source in (tmp_path / "expected").rglob("*.csv"):
        assert (
            source.read_bytes() == (actual / source.relative_to(tmp_path / "expected")).read_bytes()
        )
    for name in ("summary.json", "leakage_audit.json"):
        assert (actual / "validation" / name).read_bytes() == (
            tmp_path / "expected" / "validation" / name
        ).read_bytes()
    context = json.loads((actual / "run_context.json").read_text())
    assert (
        context["input_checks"]["effective_config_sha256"]
        == prepared.describe()["effective_config_sha256"]
    )
    assert context["input_checks"]["input_files"] == prepared.identities
    prepared.verify()


@pytest.mark.parametrize("payload", ["false", "0", "[]", "a: ["])
def test_malformed_configuration_is_not_success(tmp_path, payload, capsys):
    path = tmp_path / "invalid.yaml"
    path.write_text(payload)
    assert cli.main(["preflight", "--config", str(path)]) == 2
    assert not capsys.readouterr().out
    assert set(tmp_path.iterdir()) == {path}


@pytest.mark.parametrize("kind", [None, True, "real", "pit_certified", []])
def test_source_declaration_is_not_guessed(config_file, kind):
    raw = yaml.safe_load(config_file.read_text())
    raw["source"] = {"evidence_kind": kind}
    save_config(config_file, raw)
    with pytest.raises(ValueError, match="evidence_kind"):
        prepare_inputs(config_file)


def test_old_source_mode_remains_unspecified(config_file):
    raw = yaml.safe_load(config_file.read_text())
    raw["source"] = {"mode": "real-retrospective-price-only"}
    save_config(config_file, raw)
    assert prepare_inputs(config_file).preflight()["evidence_kind"] == "unspecified"


def test_short_history_preflight_rejects_but_run_keeps_blocked_evidence(config_file, capsys):
    raw = yaml.safe_load(config_file.read_text())
    raw["validation"]["train_size"] = 10000
    save_config(config_file, raw)
    assert cli.main(["preflight", "--config", str(config_file)]) == 2
    assert "insufficient history" in capsys.readouterr().err
    output = config_file.parent / "blocked"
    assert cli.main(["run", "--config", str(config_file), "--out", str(output)]) == 2
    assert "insufficient_history" in capsys.readouterr().err
    assert not (output / "position_scale.json").exists()
    assert json.loads((output / "decision.json").read_text())["action"] == "blocked"
    validate_standard_run(output)


def test_static_pass_does_not_bypass_latest_warmup_gate(config_file, capsys):
    raw = yaml.safe_load(config_file.read_text())
    raw["position"]["vol_lookback"] = 10000
    save_config(config_file, raw)
    assert prepare_inputs(config_file).preflight()["software_preflight"] == "pass"
    output = config_file.parent / "warmup"
    assert cli.main(["run", "--config", str(config_file), "--out", str(output)]) == 2
    assert "warmup" in capsys.readouterr().err
    assert not (output / "position_scale.json").exists()
    validate_standard_run(output)


def test_incomplete_macro_retains_hold_previous(config_file):
    context = config_file.parent / "macro.json"
    context.write_text(json.dumps({"complete": False, "values": {}, "unavailable": {}}))
    raw = yaml.safe_load(config_file.read_text())
    raw["macro"] = {"context": "macro.json", "incomplete_policy": "hold_previous"}
    save_config(config_file, raw)
    assert prepare_inputs(config_file).preflight()["software_preflight"] == "pass"
    output = config_file.parent / "held"
    assert cli.main(["run", "--config", str(config_file), "--out", str(output)]) == 0
    assert json.loads((output / "decision.json").read_text())["action"] == "hold_previous"
    assert not (output / "position_scale.json").exists()


def test_all_declared_sources_are_identified(config_file):
    root = config_file.parent
    prices = sample_prices()
    index = prices.index
    panel = pd.DataFrame({"peg": 2.0, "value_growth": 0.1, "IF": 100.0}, index=index)
    panel.to_csv(root / "signals.csv", index_label="date")
    (prices * 100).to_csv(root / "amounts.csv", index_label="date")
    pd.DataFrame({"position_scale": 0.8}, index=index).to_csv(
        root / "scale.csv", index_label="date"
    )
    (root / "macro.json").write_text(json.dumps({"complete": True, "values": {}}))
    pd.DataFrame(
        {"date": ["2020-01-01"], "series": ["x"], "value": [1], "available_at": ["2020-01-02"]}
    ).to_csv(root / "macro.csv", index=False)
    raw = yaml.safe_load(config_file.read_text())
    raw.update(
        {
            "signals": {"path": "signals.csv"},
            "activity": {"path": "amounts.csv"},
            "regime": {"combine": "cap", "history": "scale.csv"},
            "macro": {"context": "macro.json", "history": "macro.csv"},
            "futures": {"path": "signals.csv"},
            "overlay": {"mode": "futures", "contracts": ["IF"], "margin_rate": 0.1},
            "valuation": {
                "column": "peg",
                "lookback": 20,
                "expensive_percentile": 0.2,
                "scale_cap": 0.6,
            },
        }
    )
    save_config(config_file, raw)
    prepared = prepare_inputs(config_file)
    assert set(prepared.preflight()["input_files"]) == {
        "config",
        "input.path",
        "signals.path",
        "activity.path",
        "regime.history",
        "macro.context",
        "macro.history",
        "futures.path",
    }
    assert prepared.extras["futures_prices"].equals(panel)
    pd.testing.assert_frame_equal(prepared.prices, prices, check_freq=False, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("missing", ["market", "style", "signal", "futures", "cash"])
def test_missing_native_inputs_rejected(config_file, missing):
    raw = yaml.safe_load(config_file.read_text())
    if missing == "market":
        raw["market"] = "absent"
    elif missing == "style":
        raw["style"]["pairs"][0]["left"] = "absent"
    elif missing == "signal":
        raw["style"]["pairs"][0]["signal"] = "revision"
    elif missing == "futures":
        raw["overlay"] = {"mode": "futures", "contracts": ["IF"], "margin_rate": 0.1}
    else:
        raw["cash"] = {"mode": "price", "yield_column": "absent"}
    save_config(config_file, raw)
    with pytest.raises(ValueError, match="missing"):
        prepare_inputs(config_file)


@pytest.mark.parametrize("phase", ["hash", "load", "research"])
def test_source_change_prevents_success(config_file, monkeypatch, phase):
    path = config_file.parent / "prices.csv"

    def change():
        path.write_bytes(path.read_bytes() + b"\n")

    if phase == "hash":
        original = preflight.file_sha256

        def digest(source):
            value = original(source)
            if source == path:
                change()
            return value

        monkeypatch.setattr(preflight, "file_sha256", digest)
    elif phase == "load":
        original = preflight.load_market

        def loader(*args, **kwargs):
            value = original(*args, **kwargs)
            change()
            return value

        monkeypatch.setattr(preflight, "load_market", loader)
    else:
        original = cli.run_study

        def research(*args, **kwargs):
            value = original(*args, **kwargs)
            change()
            return value

        monkeypatch.setattr(cli, "run_study", research)
    output = config_file.parent / "result"
    assert cli.main(["run", "--config", str(config_file), "--out", str(output)]) == 2
    assert not output.exists()


def test_output_cannot_contain_inputs(config_file, monkeypatch):
    monkeypatch.setattr(cli, "run_study", lambda *args, **kwargs: pytest.fail("must reject first"))
    before = config_file.read_bytes()
    assert cli.main(["run", "--config", str(config_file), "--out", str(config_file.parent)]) == 2
    assert config_file.read_bytes() == before


def test_paths_still_resolve_from_original_config(config_file, tmp_path, monkeypatch):
    nested = tmp_path / "other"
    nested.mkdir()
    monkeypatch.chdir(nested)
    prepared = prepare_inputs(config_file)
    assert prepared.sources["input.path"] == config_file.parent / "prices.csv"


@pytest.mark.parametrize("section", ["signals", "activity", "futures", "macro", "regime"])
def test_optional_null_sources_keep_existing_semantics(config_file, section):
    raw = yaml.safe_load(config_file.read_text())
    raw[section] = None
    save_config(config_file, raw)
    assert prepare_inputs(config_file).preflight()["software_preflight"] == "pass"


@pytest.mark.parametrize("section", ["input", "signals", "activity", "futures"])
def test_non_mapping_source_is_a_clear_input_error(config_file, section, capsys):
    raw = yaml.safe_load(config_file.read_text())
    raw[section] = ["bad"]
    save_config(config_file, raw)
    assert cli.main(["preflight", "--config", str(config_file)]) == 2
    assert f"{section} must be a mapping" in capsys.readouterr().err


def test_snapshot_and_date_selection_are_shared(config_file):
    prices = sample_prices()
    raw = yaml.safe_load(config_file.read_text())
    raw["input"]["start"] = str(prices.index[10].date())
    raw["input"]["end"] = str(prices.index[-10].date())
    raw["regime"] = {"combine": "cap", "snapshot": "scale.json"}
    (config_file.parent / "scale.json").write_text('{"position_scale": 0.7}')
    save_config(config_file, raw)
    prepared = prepare_inputs(config_file)
    assert len(prepared.prices) == len(prices) - 19
    assert prepared.extras["regime_snapshot"] == 0.7
    assert "regime.snapshot" in prepared.preflight()["input_files"]


def test_context_write_failure_does_not_publish_scale(config_file, monkeypatch):
    import quant_timing.export as export

    original = export._write_json

    def fail_context(path, payload):
        if path.name == "run_context.json":
            raise OSError("context failed")
        return original(path, payload)

    monkeypatch.setattr(export, "_write_json", fail_context)
    output = config_file.parent / "failed-export"
    assert cli.main(["run", "--config", str(config_file), "--out", str(output)]) == 2
    assert not output.exists()


@pytest.mark.parametrize("factor", [0, 1, 2, 10])
def test_cost_override_preserves_recipe_and_source_identity(config_file, factor):
    raw = yaml.safe_load(config_file.read_text())
    raw["costs"]["impact_coef"] = 0.2
    save_config(config_file, raw)
    original = config_file.read_bytes()
    override = config_file.parent / "override.json"
    override.write_text(json.dumps({"cost_multiplier": factor}))
    prepared = prepare_inputs(config_file, override)
    assert prepared.raw["costs"] == {"bps": 10 * factor, "impact_coef": 0.2 * factor}
    assert prepared.raw["position"] == raw["position"]
    assert prepared.raw["validation"] == raw["validation"]
    assert prepared.sources["overrides"] == override
    assert config_file.read_bytes() == original
    prepared.verify()


@pytest.mark.parametrize("value", [-1, 11, True, "2", float("nan"), float("inf")])
def test_cost_override_rejects_invalid_values(config_file, value):
    override = config_file.parent / "override.json"
    override.write_text(json.dumps({"cost_multiplier": value}))
    with pytest.raises(ValueError, match="cost_multiplier"):
        prepare_inputs(config_file, override)


def test_hashed_metrics_cover_folds_context_and_publication(config_file):
    output = config_file.parent / "protected"
    assert cli.main(["run", "--config", str(config_file), "--out", str(output)]) == 0
    validate_standard_run(output)
    metrics = json.loads((output / "standard/metrics.json").read_text(encoding="utf-8"))
    assert metrics["publication"] == json.loads(
        (output / "decision.json").read_text(encoding="utf-8")
    )
    assert metrics["input_context"] == json.loads(
        (output / "run_context.json").read_text(encoding="utf-8")
    )
    assert metrics["fold_metrics_sha256"] == preflight.file_sha256(
        output / "validation/fold_metrics.csv"
    )
    assert len(metrics["walk_forward_folds"]) == metrics["fold_count"]
    metrics["publication"]["position_scale"] = 0.12345
    (output / "standard/metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="mutated"):
        validate_standard_run(output)
