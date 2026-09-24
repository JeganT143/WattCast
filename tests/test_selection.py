import copy
import itertools
import json
from pathlib import Path

import pytest

from src.tuning.selection import load_val_reference, score_configurations, select_configuration

LR_MAE, LR_RMSE = 40.0, 80.0


def _row(epochs, delta, mae, rmse):
    return {"max_epochs": epochs, "huber_delta": delta, "mean_val_mae": mae, "mean_val_rmse": rmse}


def _key(row):
    return (row["max_epochs"], row["huber_delta"])


def test_score_configurations_computes_the_two_ratios_and_their_max():
    (r,) = score_configurations([_row(10, 20.0, 30.0, 100.0)], LR_MAE, LR_RMSE)
    assert r["val_mae_ratio"] == 0.75
    assert r["val_rmse_ratio"] == 1.25
    assert r["selection_score"] == 1.25
    assert _key(r) == (10, 20.0)
    assert r["mean_val_mae"] == 30.0 and r["mean_val_rmse"] == 100.0


def test_score_configurations_preserves_order_and_does_not_mutate_the_input():
    rows = [_row(10, 20.0, 30.0, 100.0), _row(20, 40.0, 50.0, 60.0)]
    before = copy.deepcopy(rows)
    scored = score_configurations(rows, LR_MAE, LR_RMSE)
    assert rows == before
    assert [_key(r) for r in scored] == [(10, 20.0), (20, 40.0)]


def test_lowest_selection_score_wins_not_the_best_single_metric():
    rows = [
        _row(10, 20.0, 20.0, 120.0),  # ratios 0.5 / 1.5 -> score 1.5 (best MAE ratio, worst RMSE ratio)
        _row(20, 20.0, 40.0, 80.0),   # ratios 1.0 / 1.0 -> score 1.0
        _row(35, 20.0, 44.0, 72.0),   # ratios 1.1 / 0.9 -> score 1.1
    ]
    chosen = select_configuration(rows, LR_MAE, LR_RMSE)
    assert _key(chosen) == (20, 20.0)
    assert chosen["selection_score"] == 1.0


def test_tie_on_score_goes_to_the_lower_mae_ratio():
    a = _row(10, 20.0, 20.0, 80.0)  # 0.5 / 1.0 -> score 1.0
    b = _row(20, 20.0, 40.0, 60.0)  # 1.0 / 0.75 -> score 1.0
    for rows in ([a, b], [b, a]):
        assert _key(select_configuration(rows, LR_MAE, LR_RMSE)) == (10, 20.0)


def test_tie_on_score_and_mae_ratio_goes_to_the_lower_rmse_ratio():
    a = _row(10, 20.0, 40.0, 40.0)  # 1.0 / 0.5
    b = _row(20, 20.0, 40.0, 60.0)  # 1.0 / 0.75
    for rows in ([a, b], [b, a]):
        assert _key(select_configuration(rows, LR_MAE, LR_RMSE)) == (10, 20.0)


def test_full_tie_on_all_three_ratios_goes_to_the_lower_max_epochs_even_with_a_higher_delta():
    a = _row(35, 20.0, 40.0, 80.0)
    b = _row(10, 40.0, 40.0, 80.0)
    for rows in ([a, b], [b, a]):
        assert _key(select_configuration(rows, LR_MAE, LR_RMSE)) == (10, 40.0)


def test_tie_including_max_epochs_goes_to_the_lower_huber_delta():
    a = _row(20, 60.0, 40.0, 80.0)
    b = _row(20, 20.0, 40.0, 80.0)
    for rows in ([a, b], [b, a]):
        assert _key(select_configuration(rows, LR_MAE, LR_RMSE)) == (20, 20.0)


def test_selection_does_not_depend_on_input_order():
    rows = [
        _row(10, 20.0, 40.0, 80.0),
        _row(10, 40.0, 40.0, 80.0),
        _row(20, 20.0, 20.0, 80.0),
        _row(20, 40.0, 40.0, 60.0),
        _row(35, 60.0, 44.0, 72.0),
        _row(50, 60.0, 30.0, 100.0),
    ]
    expected = select_configuration(rows, LR_MAE, LR_RMSE)
    assert _key(expected) == (20, 20.0)
    for perm in itertools.permutations(rows):
        assert select_configuration(list(perm), LR_MAE, LR_RMSE) == expected


def test_there_is_no_tolerance_band():
    just_worse = _row(10, 20.0, 40.0, 80.0 * (1 + 1e-12))  # score a hair above 1.0
    exact = _row(50, 60.0, 40.0, 80.0)                      # score exactly 1.0, but higher epochs and delta
    chosen = select_configuration([just_worse, exact], LR_MAE, LR_RMSE)
    assert _key(chosen) == (50, 60.0)


def test_empty_rows_raise():
    with pytest.raises(ValueError, match="empty"):
        select_configuration([], LR_MAE, LR_RMSE)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["mean_val_mae", "mean_val_rmse"])
def test_non_finite_metrics_raise(field, bad):
    row = _row(10, 20.0, 40.0, 80.0)
    row[field] = bad
    with pytest.raises(ValueError, match="non-finite"):
        select_configuration([row], LR_MAE, LR_RMSE)


@pytest.mark.parametrize("ref", [(0.0, 80.0), (40.0, 0.0), (-1.0, 80.0), (40.0, float("nan"))])
def test_invalid_references_raise(ref):
    with pytest.raises(ValueError, match="reference"):
        select_configuration([_row(10, 20.0, 40.0, 80.0)], *ref)


def test_duplicate_configurations_raise():
    with pytest.raises(ValueError, match="duplicate"):
        select_configuration([_row(10, 20.0, 40.0, 80.0), _row(10, 20.0, 30.0, 70.0)], LR_MAE, LR_RMSE)


def test_missing_keys_raise():
    row = _row(10, 20.0, 40.0, 80.0)
    del row["mean_val_rmse"]
    with pytest.raises(ValueError, match="missing"):
        select_configuration([row], LR_MAE, LR_RMSE)


def _golden(tmp_path):
    payload = {
        "meta": {},
        "results": {
            "linear_regression|h6": {
                "metrics": {"train_mae": 7.0, "val_mae": 1.5, "val_rmse": 2.5, "test_mae": 111.0, "test_rmse": 222.0}
            },
            "linear_regression|h1": {
                "metrics": {"train_mae": 7.0, "val_mae": 9.0, "val_rmse": 9.5, "test_mae": 999.0, "test_rmse": 888.0}
            },
        },
    }
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(payload))
    return path


def test_load_val_reference_returns_only_the_h6_validation_values(tmp_path):
    assert load_val_reference(_golden(tmp_path)) == {"lr_val_mae": 1.5, "lr_val_rmse": 2.5}


def test_load_val_reference_raises_when_the_entry_is_missing(tmp_path):
    path = tmp_path / "golden.json"
    path.write_text(json.dumps({"meta": {}, "results": {}}))
    with pytest.raises(ValueError, match="linear_regression"):
        load_val_reference(path)


def test_load_val_reference_reads_the_real_fixture_unrounded():
    path = Path(__file__).resolve().parent / "fixtures" / "golden_baselines.json"
    if not path.exists():
        pytest.skip("golden fixture not present")
    ref = load_val_reference(path)
    metrics = json.loads(path.read_text())["results"]["linear_regression|h6"]["metrics"]
    assert set(ref) == {"lr_val_mae", "lr_val_rmse"}
    assert ref["lr_val_mae"] == metrics["val_mae"]
    assert ref["lr_val_rmse"] == metrics["val_rmse"]
    assert ref["lr_val_mae"] > 0 and ref["lr_val_rmse"] > 0
