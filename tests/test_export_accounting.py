from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from quant_timing import export
from quant_timing.book import simulate
from quant_timing.cli import main
from quant_timing.contract import validate_standard_run
from quant_timing.study import run_study


def config():
    return {
        "market": "market",
        "position": {"model": "tsmom", "return_window": 1, "floor": 0.5, "cap": 0.5},
        "overlay": {"mode": "futures", "contracts": ["IF"], "margin_rate": 0.1, "beta_window": 2},
        "costs": {"bps": 10.0},
        "validation": {"train_size": 2, "test_size": 2, "step_size": 2, "embargo": 0},
    }


@pytest.mark.parametrize("separate_futures", [False, True])
def test_futures_cli_exports_complete_valued_book_before_publishing(tmp_path, separate_futures):
    dates = pd.bdate_range("2026-09-01", periods=12)
    prices = pd.DataFrame({"date": dates, "market": range(100, 112)})
    raw = config()
    if separate_futures:
        future_path = tmp_path / "futures.csv"
        pd.DataFrame({"date": dates, "IF": range(100, 112)}).to_csv(future_path, index=False)
        raw["futures"] = {"path": str(future_path)}
    else:
        prices["IF"] = range(100, 112)
    price_path = tmp_path / "prices.csv"
    prices.to_csv(price_path, index=False)
    raw["input"] = {"path": str(price_path)}
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(raw))
    out = tmp_path / "run"
    assert main(["run", "--config", str(cfg), "--out", str(out)]) == 0
    manifest = validate_standard_run(out)
    assert "valuation_prices" in manifest.dataset_snapshots
    assert manifest.tags["cost_unit"] == "currency"
    assert (out / "position_scale.json").is_file()
    positions = pd.read_csv(out / "standard/positions.csv")
    active = positions[positions["symbol"] == "IF"]
    assert active["quantity"].lt(0).any()
    margin = positions[positions["symbol"] == "MARGIN"]
    assert margin["quantity"].equals(margin["market_value"])
    assert positions[["quantity", "market_value", "weight"]].notna().all().all()
    funded = (
        positions[~positions.symbol.isin(["IF", "IC", "IM"])].groupby("date").market_value.sum()
    )
    nav = pd.read_csv(out / "standard/returns.csv").set_index("date").nav
    assert funded.tolist() == pytest.approx(nav.loc[funded.index].tolist())


@pytest.mark.parametrize("failure_stage", ["write_standard_run", "validate_standard_run"])
def test_failed_validation_publishes_no_run_or_scale(tmp_path, monkeypatch, failure_stage):
    dates = pd.bdate_range("2026-09-01", periods=12)
    prices = pd.DataFrame({"market": range(100, 112)}, index=dates)
    raw = config()
    raw.pop("overlay")
    result = run_study(prices, raw)
    out = tmp_path / "run"

    def reject(*args, **kwargs):
        assert not (out / "position_scale.json").exists()
        raise ValueError("injected artifact validation failure")

    monkeypatch.setattr(export, failure_stage, reject)
    with pytest.raises(ValueError, match="artifact validation failure"):
        export.write_run(result, prices, out)
    assert not out.exists()


def test_export_preserves_existing_files_when_destination_changes_during_staging(
    tmp_path, monkeypatch
):
    dates = pd.bdate_range("2026-09-01", periods=12)
    prices = pd.DataFrame({"market": range(100, 112)}, index=dates)
    raw = config()
    raw.pop("overlay")
    result = run_study(prices, raw)
    out = tmp_path / "run"
    out.mkdir()
    original_validate = export.validate_standard_run

    def occupy(staged):
        original_validate(staged)
        (out / "existing.txt").write_text("preserve me")

    monkeypatch.setattr(export, "validate_standard_run", occupy)
    with pytest.raises(FileExistsError, match="nonempty"):
        export.write_run(result, prices, out)
    assert (out / "existing.txt").read_text() == "preserve me"
    assert not (out / "position_scale.json").exists()
    assert not (out / "standard").exists()


def test_missing_active_valuation_fails_before_publication(tmp_path):
    dates = pd.bdate_range("2026-09-01", periods=12)
    prices = pd.DataFrame({"market": range(100, 112)}, index=dates)
    futures = pd.DataFrame({"IF": range(100, 112)}, index=dates)
    result = run_study(prices, config(), futures_prices=futures)
    result.valuation_prices.loc[dates[2], "IF"] = float("nan")
    with pytest.raises(ValueError, match="valuation price for IF"):
        export.write_run(result, prices, tmp_path / "run")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("bps", [0.0, 100.0])
def test_pretrade_positions_and_orders_conserve_units_through_full_exit(bps):
    dates = pd.bdate_range("2026-09-01", periods=4)
    prices = pd.DataFrame({"A": [100.0, 110.0, 110.0, 110.0], "B": 100.0}, index=dates)
    weights = pd.DataFrame(
        {"A": [0.5, 0.0, 0.0, 0.5], "B": [0.5, 1.0, 1.0, 0.5], "CASH": 0.0}, index=dates
    )
    ledger = simulate(weights, weights, prices, "A", bps)
    frames = export._standard_frames(
        SimpleNamespace(
            weights=weights,
            realized=ledger,
            position=pd.DataFrame({"position_scale": 1.0}, index=dates),
        ),
        prices,
    )
    pos = frames["positions"].set_index(["date", "symbol"])
    assert pos.loc[("2026-09-02", "A"), "quantity"] == pytest.approx(0.005)
    assert pos.loc[("2026-09-02", "A"), "market_value"] == pytest.approx(0.55)
    assert pos.loc[("2026-09-02", "A"), "return_weight"] == 0.5
    assert pos.loc[("2026-09-02", "A"), "weight"] == pytest.approx(0.55 / 1.05)
    for date in dates[1:]:
        day = str(date.date())
        orders = frames["orders"].query("timestamp == @day")
        for symbol in weights.columns:
            previous = pos.loc[(day, symbol), "quantity"]
            changes = orders[orders.symbol == symbol]
            delta = sum(
                row.quantity * (1 if row.side == "buy" else -1) for row in changes.itertuples()
            )
            price = 1.0 if symbol == "CASH" else prices.loc[date, symbol]
            target = weights.loc[date, symbol] * ledger.loc[date, "nav"] / price
            assert previous + delta == pytest.approx(target, abs=1e-12)
