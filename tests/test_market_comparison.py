from pathlib import Path

import pandas as pd

from quant_timing.cli import main
from quant_timing.io import load_prices


def test_public_snapshot_covers_the_named_sleeves() -> None:
    root = Path(__file__).resolve().parents[1]
    prices = load_prices(root / "data" / "public" / "indices.csv")
    required = {
        "hs300",
        "csi500",
        "csi1000",
        "csi300_value",
        "csi300_growth",
        "cni_value",
        "cni_growth",
        "dividend",
        "cyclical",
        "bank",
        "broker",
        "pharma",
        "electronics",
        "bond_etf",
    }
    assert required <= set(prices.columns)
    assert len(prices) > 1500
    futures = pd.read_csv(root / "data" / "public" / "futures.csv")
    assert {"IF", "IC", "IM"} <= set(futures.columns)
    assert futures.loc[futures["date"] < "2022-07-22", "IM"].isna().all()


def test_market_comparison_completes_out_of_sample(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    code = main(
        [
            "compare",
            "--config",
            str(root / "configs" / "market_comparison.yaml"),
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    table = pd.read_csv(tmp_path / "model_comparison.csv")
    assert set(table["book"]) == {"position_hs300", "style", "futures_hs300"}
    assert set(table["model"]) == {"rules", "vol_target", "tsmom", "moving_average"}
    assert table["scored_folds"].min() >= 1
    assert table["leakage_passed"].all()
    assert table["mean_excess_return"].notna().all()
    assert table["full_sample_drawdown"].notna().all()
    assert table["mean_turnover"].notna().all()
    assert table["capacity"].isna().all()
    assert table["median_participation"].isna().all()
    assert set(table["sample_start"]) == {"2019-03-08"}


def test_volume_cannot_be_used_with_monetary_capital():
    import pytest
    from quant_timing.config import load_yaml, resolve_config

    root = Path(__file__).resolve().parents[1]
    raw = load_yaml(root / "configs" / "market_comparison.yaml")
    raw["position"] = {"model": "rules", **raw["models"]["rules"]}
    raw["costs"]["capital"] = 100000000
    with pytest.raises(ValueError, match="activity.unit: CNY"):
        resolve_config(raw)
