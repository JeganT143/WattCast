"""Tests for src.evaluation.walk_forward_verdict — the pure verdict aggregator
for the registered Phase 5 walk-forward comparison rule (DECISIONS.md,
"Phase 5: walk-forward comparison rule (pre-registration)")."""

import copy
import dataclasses
import math

import pytest

from config.walk_forward import (
    BETTER_MARGIN,
    EVAL_DAYS,
    FIRST_EVAL_START,
    FREQ_MINUTES,
    N_FOLDS,
    NEURAL_SEEDS,
    WORSE_MARGIN,
)
from src.evaluation.walk_forward_verdict import (
    WalkForwardVerdict,
    mean_ratios,
    walk_forward_verdict,
)


def same_for_all_seeds(value, n=8):
    return {seed: [value] * n for seed in NEURAL_SEEDS}


def with_seed_override(base, seed, values):
    out = {k: list(v) for k, v in base.items()}
    out[seed] = list(values)
    return out


def test_all_better_yields_shown_better():
    ratios = same_for_all_seeds(0.95)
    result = walk_forward_verdict(ratios, ratios)
    assert result.verdict == "shown better"
    for m in result.mean_mae_ratios:
        assert math.isclose(m, 0.95, rel_tol=0, abs_tol=1e-12)
    assert result.mean_mae_ratios == result.mean_rmse_ratios


def test_better_boundary_inclusive_and_just_past_is_not_shown():
    at_boundary = same_for_all_seeds(0.99)
    result = walk_forward_verdict(at_boundary, at_boundary)
    assert result.verdict == "shown better"

    just_past = same_for_all_seeds(math.nextafter(0.99, 2.0))
    result_past = walk_forward_verdict(just_past, just_past)
    assert result_past.verdict == "not shown"


def test_every_seed_required_one_seed_off_is_not_shown():
    mae = same_for_all_seeds(0.95)
    rmse = with_seed_override(same_for_all_seeds(0.95), 44, [1.0] * 8)
    result = walk_forward_verdict(mae, rmse)
    assert result.verdict == "not shown"


def test_mixed_metrics_is_not_shown():
    mae = same_for_all_seeds(0.95)
    rmse = same_for_all_seeds(1.02)
    result = walk_forward_verdict(mae, rmse)
    assert result.verdict == "not shown"


def test_all_worse_and_boundary_and_just_short():
    worse = same_for_all_seeds(1.05)
    assert walk_forward_verdict(worse, worse).verdict == "shown worse"

    at_boundary = same_for_all_seeds(1.01)
    assert walk_forward_verdict(at_boundary, at_boundary).verdict == "shown worse"

    just_short = same_for_all_seeds(math.nextafter(1.01, 0.0))
    assert walk_forward_verdict(just_short, just_short).verdict == "not shown"


def test_mean_not_median_or_worst_fold():
    folds = [0.9, 0.9, 0.9, 0.9, 0.9, 1.3, 1.3, 1.3]
    expected_mean = math.fsum(folds) / len(folds)
    assert math.isclose(expected_mean, 1.05, rel_tol=0, abs_tol=1e-12)

    ratios = {seed: list(folds) for seed in NEURAL_SEEDS}
    result = walk_forward_verdict(ratios, ratios)
    for m in result.mean_mae_ratios:
        assert math.isclose(m, expected_mean, rel_tol=0, abs_tol=1e-12)
    # Mean (1.05) is >= WORSE_MARGIN (1.01) for every seed and both metrics,
    # so the correct mean-based verdict is "shown worse" — using the median
    # (0.9) or picking the worst fold in isolation would not reproduce this.
    assert result.verdict == "shown worse"

    reordered = [1.3, 1.3, 1.3, 0.9, 0.9, 0.9, 0.9, 0.9]
    ratios_reordered = {seed: list(reordered) for seed in NEURAL_SEEDS}
    result_reordered = walk_forward_verdict(ratios_reordered, ratios_reordered)
    assert result_reordered.mean_mae_ratios == result.mean_mae_ratios
    assert result_reordered.mean_rmse_ratios == result.mean_rmse_ratios


def test_sensitivity_first_k_does_not_change_verdict():
    folds = [0.9, 0.9, 0.9, 0.9, 0.9, 1.3, 1.3, 1.3]
    ratios = {seed: list(folds) for seed in NEURAL_SEEDS}

    expected_first6 = (0.9 * 5 + 1.3) / 6
    sensitivity = mean_ratios(ratios, first_k=6)
    for m in sensitivity:
        assert math.isclose(m, expected_first6, rel_tol=0, abs_tol=1e-12)

    result = walk_forward_verdict(ratios, ratios)
    assert result.verdict == "shown worse"
    for m in result.mean_mae_ratios:
        assert math.isclose(m, 1.05, rel_tol=0, abs_tol=1e-12)


def test_validation_raises_value_error():
    good = same_for_all_seeds(0.95)

    missing_seed = {k: v for k, v in good.items() if k != 44}
    with pytest.raises(ValueError):
        walk_forward_verdict(missing_seed, good)

    extra_seed = dict(good)
    extra_seed[45] = [0.95] * 8
    with pytest.raises(ValueError):
        walk_forward_verdict(extra_seed, good)

    seven_ratios = with_seed_override(good, 42, [0.95] * 7)
    with pytest.raises(ValueError):
        walk_forward_verdict(seven_ratios, good)

    nan_ratios = with_seed_override(good, 42, [0.95] * 7 + [float("nan")])
    with pytest.raises(ValueError):
        walk_forward_verdict(nan_ratios, good)

    inf_ratios = with_seed_override(good, 42, [0.95] * 7 + [float("inf")])
    with pytest.raises(ValueError):
        walk_forward_verdict(inf_ratios, good)

    zero_ratios = with_seed_override(good, 42, [0.95] * 7 + [0.0])
    with pytest.raises(ValueError):
        walk_forward_verdict(zero_ratios, good)

    negative_ratios = with_seed_override(good, 42, [0.95] * 7 + [-0.1])
    with pytest.raises(ValueError):
        walk_forward_verdict(negative_ratios, good)

    mismatched_keys_rmse = {k: v for k, v in good.items() if k != 44}
    mismatched_keys_rmse[45] = [0.95] * 8
    with pytest.raises(ValueError):
        walk_forward_verdict(good, mismatched_keys_rmse)

    with pytest.raises(ValueError):
        mean_ratios(good, first_k=0)

    with pytest.raises(ValueError):
        mean_ratios(good, first_k=9)


def test_purity_and_frozen_result():
    mae = same_for_all_seeds(0.95)
    rmse = same_for_all_seeds(0.95)
    mae_before = copy.deepcopy(mae)
    rmse_before = copy.deepcopy(rmse)

    result = walk_forward_verdict(mae, rmse)

    assert mae == mae_before
    assert rmse == rmse_before

    assert isinstance(result, WalkForwardVerdict)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.verdict = "shown better"

    assert len(result.mean_mae_ratios) == 3
    assert len(result.mean_rmse_ratios) == 3


def test_appended_and_existing_constants():
    assert NEURAL_SEEDS == (42, 43, 44)
    assert BETTER_MARGIN == 0.99
    assert WORSE_MARGIN == 1.01

    assert FIRST_EVAL_START == "2016-03-01 00:00"
    assert N_FOLDS == 8
    assert EVAL_DAYS == 7
    assert FREQ_MINUTES == 10
