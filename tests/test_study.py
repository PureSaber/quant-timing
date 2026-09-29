from __future__ import annotations

import pandas as pd
import pytest

from quant_timing.config import resolve_config
from quant_timing.study import run_study
from tests.support import sample_prices, timing_config


def test_books_sum_to_one_and_cash_is_the_residual() -> None:
    result = run_study(sample_prices(), timing_config(), audit=False)
    assert result.weights.sum(axis=1).sub(1).abs().max() <= 1e-8
    residual = (result.weights["CASH"] - (1.0 - result.position["position_scale"])).abs()
    assert residual.max() <= 1e-8
    assert result.leakage["reason"] == "not_run"
    assert result.summary["status"] == "audit_not_run"


def test_truncation_and_a_future_price_do_not_change_past_weights() -> None:
    prices = sample_prices()
    config = timing_config()
    full = run_study(prices, config, audit=False)
    stamp = prices.index[200]
    prefix = run_study(prices.loc[:stamp], config, audit=False)
    pd.testing.assert_series_equal(prefix.weights.loc[stamp], full.weights.loc[stamp])

    revised = prices.copy()
    revised.iloc[-1, revised.columns.get_loc("small")] *= 1.5
    changed = run_study(revised, config, audit=False)
    earlier = prices.index[-2]
    pd.testing.assert_series_equal(changed.weights.loc[earlier], full.weights.loc[earlier])


def test_walk_forward_scores_only_after_the_embargo() -> None:
    prices = sample_prices()
    result = run_study(prices, timing_config(), audit=True)
    assert result.summary["status"] == "complete"
    assert result.summary["leakage_passed"] is True
    assert result.summary["scored_folds"] >= 1
    embargo = timing_config()["validation"]["embargo"]
    for row in result.folds.itertuples(index=False):
        train_loc = prices.index.get_loc(pd.Timestamp(row.train_end))
        test_loc = prices.index.get_loc(pd.Timestamp(row.test_start))
        assert test_loc - train_loc == embargo + 1
        assert row.n_decisions > 0


def test_position_only_matched_benchmark_has_no_style_excess() -> None:
    raw = timing_config()
    raw.pop("style")
    result = run_study(sample_prices(), raw, audit=False)
    assert list(result.weights.columns) == ["market", "CASH"]
    excess = result.folds["matched_excess_return"].dropna()
    assert excess.abs().max() <= 1e-10


def test_snapshot_cap_applies_to_the_latest_decision_only() -> None:
    prices = sample_prices()
    raw = timing_config()
    raw["regime"] = {"combine": "cap", "snapshot": "snapshot.json"}
    plain = run_study(prices, timing_config(), audit=False)
    capped = run_study(prices, raw, regime_snapshot=0.25, audit=False)
    assert capped.position["position_scale"].equals(plain.position["position_scale"])
    assert capped.latest["regime"] != "warmup"
    assert capped.latest["position_scale"] <= 0.25
    assert capped.latest["signals"]["macro_policy"].endswith("snapshot_cap")


def test_history_cap_is_causal_and_binds_mature_decisions() -> None:
    prices = sample_prices()
    raw = timing_config()
    raw["regime"] = {"combine": "cap", "history": "scales.csv"}
    external = pd.Series(0.2, index=prices.index)
    result = run_study(prices, raw, regime_history=external, audit=False)
    mature = result.position["regime"].ne("warmup")
    assert result.position.loc[mature, "position_scale"].max() <= 0.2
    assert (result.position.loc[mature, "model_scale"] > 0.2).any()


def test_invalid_policy_is_rejected() -> None:
    raw = timing_config()
    raw["style"]["tilt"] = 0.4
    with pytest.raises(ValueError, match="tilt"):
        resolve_config(raw)
    raw = timing_config()
    raw["style"]["pairs"][0]["group_weight"] = 0.2
    with pytest.raises(ValueError, match="sum to 1"):
        resolve_config(raw)
    raw = timing_config()
    raw["regime"] = {"combine": "cap", "history": "a.csv", "snapshot": "b.json"}
    with pytest.raises(ValueError, match="either"):
        resolve_config(raw)
