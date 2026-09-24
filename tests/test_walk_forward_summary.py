"""Tests for src.evaluation.walk_forward_summary.summarize_walk_forward — a
pure summary of walk-forward runner records into the registered
per-(neural model, reference) verdicts and descriptive Tier 2 numbers.
Synthetic records only: no data files, no models, no training."""

import json
import random
import statistics

import pytest

from config.walk_forward import NEURAL_SEEDS
from src.evaluation.walk_forward import FoldRunRecord
from src.evaluation.walk_forward_summary import summarize_walk_forward

N_FOLDS = 8


def _rec(fold, model, seed, mae, rmse, mape=10.0):
    return FoldRunRecord(
        fold=fold,
        model=model,
        seed=seed,
        mae=mae,
        rmse=rmse,
        mape=mape,
        n_evaluated=1008,
        n_warmup_rows=0,
        n_ineligible_label_rows=6,
    )


def _build_records(
    *,
    lr_mae_by_fold,
    lr_rmse_by_fold,
    rf_mae_by_fold=None,
    rf_rmse_by_fold=None,
    n1_mae_by_seed_fold,
    n1_rmse_by_seed_fold,
    mape=10.0,
):
    rf_mae_by_fold = rf_mae_by_fold if rf_mae_by_fold is not None else lr_mae_by_fold
    rf_rmse_by_fold = rf_rmse_by_fold if rf_rmse_by_fold is not None else lr_rmse_by_fold

    records = []
    for fold in range(1, N_FOLDS + 1):
        i = fold - 1
        records.append(_rec(fold, "naive_persistence", None, 50.0, 60.0, mape))
        records.append(_rec(fold, "naive_seasonal", None, 55.0, 65.0, mape))
        records.append(_rec(fold, "linear_regression", None, lr_mae_by_fold[i], lr_rmse_by_fold[i], mape))
        records.append(_rec(fold, "random_forest", None, rf_mae_by_fold[i], rf_rmse_by_fold[i], mape))
        for seed in NEURAL_SEEDS:
            records.append(
                _rec(
                    fold,
                    "n1",
                    seed,
                    n1_mae_by_seed_fold[seed][i],
                    n1_rmse_by_seed_fold[seed][i],
                    mape,
                )
            )
    return records


def _constant(value):
    return [value] * N_FOLDS


def _constant_by_seed(value):
    return {seed: _constant(value) for seed in NEURAL_SEEDS}


def test_a_both_references_shown_better():
    records = _build_records(
        lr_mae_by_fold=_constant(100.0),
        lr_rmse_by_fold=_constant(200.0),
        rf_mae_by_fold=_constant(100.0),
        rf_rmse_by_fold=_constant(200.0),
        n1_mae_by_seed_fold=_constant_by_seed(95.0),
        n1_rmse_by_seed_fold=_constant_by_seed(190.0),
    )
    result = summarize_walk_forward(records, neural_models=["n1"])

    for reference in ("linear_regression", "random_forest"):
        entry = result["verdicts"]["n1"][reference]
        assert entry["verdict"] == "shown better"
        for ratio in entry["mean_mae_ratios"]:
            assert ratio == pytest.approx(0.95, rel=1e-12)
        for ratio in entry["mean_rmse_ratios"]:
            assert ratio == pytest.approx(0.95, rel=1e-12)


def test_b_per_reference_independence():
    records = _build_records(
        lr_mae_by_fold=_constant(100.0),
        lr_rmse_by_fold=_constant(200.0),
        rf_mae_by_fold=_constant(80.0),
        rf_rmse_by_fold=_constant(160.0),
        n1_mae_by_seed_fold=_constant_by_seed(95.0),
        n1_rmse_by_seed_fold=_constant_by_seed(190.0),
    )
    result = summarize_walk_forward(records, neural_models=["n1"])

    assert result["verdicts"]["n1"]["linear_regression"]["verdict"] == "shown better"
    assert result["verdicts"]["n1"]["random_forest"]["verdict"] == "shown worse"


def test_c_mean_of_ratios_not_ratio_of_means():
    lr_mae = [100.0, 200.0] * 4
    n1_mae = [90.0, 200.0] * 4

    # The prompt that specified this test asserted the verdict would be
    # "not shown" here, then in the same breath asserted "shown better".
    # Independent arithmetic: mean MAE ratio = mean([90/100, 200/200]*4)
    # = mean([0.9, 1.0]*4) = 0.95; mean RMSE ratio = 0.95 by construction.
    # Both are <= BETTER_MARGIN (0.99) for every seed, so
    # walk_forward_verdict returns "shown better", not "not shown". The
    # "not shown" wording in the prompt was a self-contradiction; this
    # test asserts the mathematically correct verdict, "shown better".
    records = _build_records(
        lr_mae_by_fold=lr_mae,
        lr_rmse_by_fold=_constant(200.0),
        rf_mae_by_fold=lr_mae,
        rf_rmse_by_fold=_constant(200.0),
        n1_mae_by_seed_fold={seed: n1_mae for seed in NEURAL_SEEDS},
        n1_rmse_by_seed_fold=_constant_by_seed(190.0),
    )

    result = summarize_walk_forward(records, neural_models=["n1"])
    entry = result["verdicts"]["n1"]["linear_regression"]

    expected_mean_ratio = 0.95
    ratio_of_means = (sum(n1_mae) / N_FOLDS) / (sum(lr_mae) / N_FOLDS)
    assert ratio_of_means != pytest.approx(expected_mean_ratio, rel=1e-6)

    for ratio in entry["mean_mae_ratios"]:
        assert ratio == pytest.approx(expected_mean_ratio, rel=1e-12)

    assert entry["verdict"] == "shown better"


def test_d_seed_pairing_and_order():
    lr_mae = _constant(100.0)
    lr_rmse = _constant(200.0)
    n1_mae_by_seed = {42: _constant(90.0), 43: _constant(90.0), 44: _constant(105.0)}
    n1_rmse_by_seed = {42: _constant(180.0), 43: _constant(180.0), 44: _constant(210.0)}

    records = _build_records(
        lr_mae_by_fold=lr_mae,
        lr_rmse_by_fold=lr_rmse,
        n1_mae_by_seed_fold=n1_mae_by_seed,
        n1_rmse_by_seed_fold=n1_rmse_by_seed,
    )
    random.Random(0).shuffle(records)

    result = summarize_walk_forward(records, neural_models=["n1"])
    entry = result["verdicts"]["n1"]["linear_regression"]

    assert entry["mean_mae_ratios"] == pytest.approx([0.9, 0.9, 1.05], rel=1e-12)
    assert entry["mean_rmse_ratios"] == pytest.approx([0.9, 0.9, 1.05], rel=1e-12)
    assert entry["verdict"] == "not shown"


def test_e_sensitivity_does_not_affect_verdict():
    lr_mae = _constant(100.0)
    lr_rmse = _constant(200.0)
    fold_ratios = [0.9] * 6 + [1.3] * 2
    n1_mae = [r * 100.0 for r in fold_ratios]
    n1_rmse = [r * 200.0 for r in fold_ratios]

    records = _build_records(
        lr_mae_by_fold=lr_mae,
        lr_rmse_by_fold=lr_rmse,
        n1_mae_by_seed_fold={seed: n1_mae for seed in NEURAL_SEEDS},
        n1_rmse_by_seed_fold={seed: n1_rmse for seed in NEURAL_SEEDS},
    )

    result = summarize_walk_forward(records, neural_models=["n1"])
    entry = result["verdicts"]["n1"]["linear_regression"]

    for ratio in entry["mean_mae_ratios"]:
        assert ratio == pytest.approx(1.0, rel=1e-12)
    assert entry["verdict"] == "not shown"

    sensitivity = entry["sensitivity_folds_1_to_6"]
    for ratio in sensitivity["mean_mae_ratios"]:
        assert ratio == pytest.approx(0.9, rel=1e-12)
    for ratio in sensitivity["mean_rmse_ratios"]:
        assert ratio == pytest.approx(0.9, rel=1e-12)

    # Sensitivity must not change the verdict field.
    assert entry["verdict"] == "not shown"


def test_f_tier2_worst_fold_folds_won_seed_spread():
    lr_mae = _constant(100.0)
    lr_rmse = _constant(200.0)
    ratios_seed42 = [0.8, 0.9, 1.0, 1.1, 1.2, 0.7, 0.95, 1.05]
    ratios_seed43 = list(reversed(ratios_seed42))
    ratios_seed44 = [1.0] * N_FOLDS

    n1_mae_by_seed = {
        42: [r * 100.0 for r in ratios_seed42],
        43: [r * 100.0 for r in ratios_seed43],
        44: [r * 100.0 for r in ratios_seed44],
    }
    n1_rmse_by_seed = {
        42: [r * 200.0 for r in ratios_seed42],
        43: [r * 200.0 for r in ratios_seed43],
        44: [r * 200.0 for r in ratios_seed44],
    }

    records = _build_records(
        lr_mae_by_fold=lr_mae,
        lr_rmse_by_fold=lr_rmse,
        n1_mae_by_seed_fold=n1_mae_by_seed,
        n1_rmse_by_seed_fold=n1_rmse_by_seed,
    )

    result = summarize_walk_forward(records, neural_models=["n1"])
    entry = result["verdicts"]["n1"]["linear_regression"]

    assert entry["worst_fold_mae_ratio"] == pytest.approx([1.2, 1.2, 1.0], rel=1e-12)
    assert entry["worst_fold_rmse_ratio"] == pytest.approx([1.2, 1.2, 1.0], rel=1e-12)
    assert entry["folds_won_mae"] == [4, 4, 0]
    assert entry["folds_won_rmse"] == [4, 4, 0]

    expected_seed_spread = statistics.stdev(entry["mean_mae_ratios"])
    assert entry["seed_spread_mae"] == pytest.approx(expected_seed_spread, rel=1e-12)
    expected_seed_spread_rmse = statistics.stdev(entry["mean_rmse_ratios"])
    assert entry["seed_spread_rmse"] == pytest.approx(expected_seed_spread_rmse, rel=1e-12)


def test_g_per_model_means_and_n_records():
    lr_mae = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    lr_rmse = [15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0]
    rf_mae = [11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 71.0, 81.0]
    rf_rmse = [16.0, 26.0, 36.0, 46.0, 56.0, 66.0, 76.0, 86.0]
    n1_mae_by_seed = {
        42: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        43: [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        44: [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    }
    n1_rmse_by_seed = {
        42: [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
        43: [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
        44: [3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
    }

    records = _build_records(
        lr_mae_by_fold=lr_mae,
        lr_rmse_by_fold=lr_rmse,
        rf_mae_by_fold=rf_mae,
        rf_rmse_by_fold=rf_rmse,
        n1_mae_by_seed_fold=n1_mae_by_seed,
        n1_rmse_by_seed_fold=n1_rmse_by_seed,
    )

    result = summarize_walk_forward(records, neural_models=["n1"])
    means = result["per_model_means"]

    for model in (
        "n1",
        "linear_regression",
        "random_forest",
        "naive_persistence",
        "naive_seasonal",
    ):
        assert model in means

    all_n1_mae = [v for seed in NEURAL_SEEDS for v in n1_mae_by_seed[seed]]
    all_n1_rmse = [v for seed in NEURAL_SEEDS for v in n1_rmse_by_seed[seed]]
    assert means["n1"]["mean_mae"] == pytest.approx(sum(all_n1_mae) / len(all_n1_mae), rel=1e-12)
    assert means["n1"]["mean_rmse"] == pytest.approx(sum(all_n1_rmse) / len(all_n1_rmse), rel=1e-12)

    assert means["linear_regression"]["mean_mae"] == pytest.approx(sum(lr_mae) / N_FOLDS, rel=1e-12)
    assert means["linear_regression"]["mean_rmse"] == pytest.approx(sum(lr_rmse) / N_FOLDS, rel=1e-12)
    assert means["random_forest"]["mean_mae"] == pytest.approx(sum(rf_mae) / N_FOLDS, rel=1e-12)
    assert means["random_forest"]["mean_rmse"] == pytest.approx(sum(rf_rmse) / N_FOLDS, rel=1e-12)

    assert result["n_records"] == len(records)


def test_h_validation_errors():
    valid_records = _build_records(
        lr_mae_by_fold=_constant(100.0),
        lr_rmse_by_fold=_constant(200.0),
        n1_mae_by_seed_fold=_constant_by_seed(90.0),
        n1_rmse_by_seed_fold=_constant_by_seed(180.0),
    )

    with pytest.raises(ValueError):
        summarize_walk_forward([], neural_models=["n1"])

    duplicated = list(valid_records) + [valid_records[0]]
    with pytest.raises(ValueError):
        summarize_walk_forward(duplicated, neural_models=["n1"])

    missing_ref_fold = [
        r for r in valid_records if not (r.model == "linear_regression" and r.fold == 3)
    ]
    with pytest.raises(ValueError):
        summarize_walk_forward(missing_ref_fold, neural_models=["n1"])

    extra_fold = list(valid_records) + [
        _rec(9, "naive_persistence", None, 50.0, 60.0)
    ]
    with pytest.raises(ValueError):
        summarize_walk_forward(extra_fold, neural_models=["n1"])

    missing_seed = [
        r for r in valid_records if not (r.model == "n1" and r.fold == 1 and r.seed == 42)
    ]
    with pytest.raises(ValueError):
        summarize_walk_forward(missing_seed, neural_models=["n1"])

    reference_with_seed = [
        _rec(r.fold, r.model, 42, r.mae, r.rmse, r.mape)
        if (r.model == "linear_regression" and r.fold == 1)
        else r
        for r in valid_records
    ]
    with pytest.raises(ValueError):
        summarize_walk_forward(reference_with_seed, neural_models=["n1"])

    with pytest.raises(ValueError):
        summarize_walk_forward(valid_records, neural_models=["n1", "n2"])

    with pytest.raises(ValueError):
        summarize_walk_forward(
            valid_records,
            neural_models=["n1"],
            baseline_models=("n1", "naive_persistence", "naive_seasonal"),
        )

    nan_mae = [
        _rec(r.fold, r.model, r.seed, float("nan"), r.rmse, r.mape)
        if (r.model == "n1" and r.fold == 1 and r.seed == 42)
        else r
        for r in valid_records
    ]
    with pytest.raises(ValueError):
        summarize_walk_forward(nan_mae, neural_models=["n1"])

    zero_reference_mae = [
        _rec(r.fold, r.model, r.seed, 0.0, r.rmse, r.mape)
        if (r.model == "linear_regression" and r.fold == 1)
        else r
        for r in valid_records
    ]
    with pytest.raises(ValueError):
        summarize_walk_forward(zero_reference_mae, neural_models=["n1"])


def test_i_purity_and_json():
    records = _build_records(
        lr_mae_by_fold=_constant(100.0),
        lr_rmse_by_fold=_constant(200.0),
        n1_mae_by_seed_fold=_constant_by_seed(95.0),
        n1_rmse_by_seed_fold=_constant_by_seed(190.0),
    )
    before = list(records)

    result = summarize_walk_forward(records, neural_models=["n1"])

    assert records == before

    json.dumps(result, allow_nan=False, sort_keys=True)

    shuffled = list(records)
    random.Random(1).shuffle(shuffled)
    result_shuffled = summarize_walk_forward(shuffled, neural_models=["n1"])
    assert result == result_shuffled

    for reference in ("linear_regression", "random_forest"):
        assert result["verdicts"]["n1"][reference]["verdict"] in (
            "shown better",
            "shown worse",
            "not shown",
        )
