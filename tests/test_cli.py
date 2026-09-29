from __future__ import annotations

import json
from pathlib import Path

from quant_timing.cli import main
from quant_timing.synthetic import write_fixture


def test_cli_run_publishes_a_scale(tmp_path) -> None:
    write_fixture(tmp_path / "sleeves.csv")
    config = {
        "market": "market",
        "run_id": "cli",
        "input": {"path": "sleeves.csv", "format": "wide", "date_col": "date"},
        "position": {
            "vol_window": 20,
            "vol_lookback": 60,
            "vol_percentile_threshold": 0.8,
            "return_window": 20,
            "risk_off_return": -0.05,
            "high_vol_scale": 0.5,
            "risk_off_scale": 0.3,
            "risk_on_scale": 1.0,
        },
        "costs": {"bps": 10},
        "validation": {"train_size": 120, "test_size": 40, "step_size": 40, "embargo": 5},
        "style": {
            "return_window": 20,
            "tilt": 0.7,
            "threshold": 0.0,
            "pairs": [
                {"name": "size", "left": "small", "right": "large", "group_weight": 0.5},
                {"name": "value_growth", "left": "value", "right": "growth", "group_weight": 0.5},
            ],
        },
    }
    config_path = tmp_path / "combined.yaml"
    config_path.write_text(_yaml(config), encoding="utf-8")
    out = tmp_path / "out"
    assert main(["run", "--config", str(config_path), "--out", str(out)]) == 0
    payload = json.loads((out / "position_scale.json").read_text(encoding="utf-8"))
    assert 0.0 <= float(payload["position_scale"]) <= 1.0
    assert (out / "validation" / "fold_metrics.csv").is_file()
    assert (out / "standard" / "run_manifest.json").is_file()


def test_shipped_config_runs(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    fixture = root / "tests" / "fixtures" / "sleeves.csv"
    if not fixture.is_file():
        write_fixture(fixture)
    code = main(
        ["run", "--config", str(root / "configs" / "combined.yaml"), "--out", str(tmp_path / "out")]
    )
    assert code == 0
    payload = json.loads((tmp_path / "out" / "position_scale.json").read_text(encoding="utf-8"))
    assert 0.0 <= float(payload["position_scale"]) <= 1.0


def _yaml(payload: dict) -> str:
    import yaml

    return yaml.safe_dump(payload, sort_keys=False)
