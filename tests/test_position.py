from __future__ import annotations

import pandas as pd
import pytest

from quant_timing.position import apply_external_cap, classify_position, position_history
from tests.support import sample_prices, timing_config


def test_high_vol_takes_priority_over_a_losing_window() -> None:
    rules = timing_config()["position"]
    regime, scale = classify_position(0.91, -0.2, rules)
    assert regime == "high_vol"
    assert scale == 0.5


def test_risk_off_and_risk_on_follow_the_window_return() -> None:
    rules = timing_config()["position"]
    assert classify_position(0.2, -0.06, rules) == ("risk_off", 0.3)
    assert classify_position(0.2, 0.01, rules) == ("risk_on", 1.0)


def test_incomplete_features_stay_in_cash() -> None:
    rules = timing_config()["position"]
    assert classify_position(float("nan"), -0.2, rules) == ("warmup", 0.0)
    assert classify_position(0.1, None, rules) == ("warmup", 0.0)


def test_history_labels_match_the_classifier() -> None:
    rules = timing_config()["position"]
    history = position_history(sample_prices()["market"], rules)
    for row in history.itertuples(index=False):
        regime, scale = classify_position(row.vol_percentile, row.window_return, rules)
        assert row.regime == regime
        assert row.model_scale == scale
        assert row.position_scale == scale


def test_external_cap_never_uses_a_later_publication() -> None:
    rules = timing_config()["position"]
    history = position_history(sample_prices()["market"], rules)
    future_only = pd.Series([0.1], index=[history.index[-1]])
    with pytest.raises(ValueError, match="does not cover"):
        apply_external_cap(history, future_only)

    external = pd.Series(0.2, index=history.index)
    external.iloc[-1] = 0.0
    capped = apply_external_cap(history, external)
    assert capped["external_scale"].iloc[-2] == pytest.approx(0.2)
    assert capped["external_scale"].iloc[-1] == pytest.approx(0.0)
    mature = capped["regime"].ne("warmup")
    assert capped.loc[mature, "position_scale"].max() <= 0.2
    assert capped.loc[~mature, "position_scale"].eq(0).all()
