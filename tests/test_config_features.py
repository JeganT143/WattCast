"""Tests for config/features.py's derived constants, notably BASELINE_CONTEXT_LAGS."""

import pytest

from config.features import (
    BASELINE_CONTEXT_LAGS,
    SEASONAL_PERIOD,
    TARGET_HORIZONS,
    _derive_baseline_context_lags,
)


def test_baseline_context_lags_are_derived_correctly():
    result = _derive_baseline_context_lags(
        horizons=[1, 6],
        seasonal_period=144,
    )

    assert result == {
        1: 143,
        6: 138,
    }


def test_baseline_context_lags_reject_zero_horizon():
    with pytest.raises(ValueError):
        _derive_baseline_context_lags(
            horizons=[0],
            seasonal_period=144,
        )


def test_baseline_context_lags_reject_negative_horizon():
    with pytest.raises(ValueError):
        _derive_baseline_context_lags(
            horizons=[-1],
            seasonal_period=144,
        )


def test_baseline_context_lags_reject_horizon_equal_to_period():
    with pytest.raises(ValueError):
        _derive_baseline_context_lags(
            horizons=[144],
            seasonal_period=144,
        )


def test_baseline_context_lags_reject_horizon_greater_than_period():
    with pytest.raises(ValueError):
        _derive_baseline_context_lags(
            horizons=[145],
            seasonal_period=144,
        )


def test_configured_baseline_context_lags_match_project_horizons():
    assert BASELINE_CONTEXT_LAGS == {
        1: 143,
        6: 138,
    }

    assert SEASONAL_PERIOD == 144
    assert TARGET_HORIZONS == [1, 6]
