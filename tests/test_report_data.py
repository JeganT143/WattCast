"""Tests for src/ui/report_data.py: every loader must return exactly what
the recorded result files contain. Where practical, a test re-reads the
source file itself (independently of the function under test) and compares.
"""

import json

import pandas as pd

from config.features import FEATURE_COLUMNS, LAG_STEPS, ROLLING_WINDOWS, SCALED_COLUMNS
from config.paths import RESULTS_DIR
from src.ui.report_data import (
    MODEL_LABELS,
    available_images,
    load_bundle_schemas,
    load_feature_config,
    load_per_fold_table,
    load_tier1_summary,
    load_verdicts,
    load_walk_forward_leaderboard,
)

WALK_FORWARD_SUMMARY_PATH = RESULTS_DIR / "walk_forward_summary.json"
WALK_FORWARD_RECORDS_PATH = RESULTS_DIR / "walk_forward_records.jsonl"
FINAL_EVALUATION_PATH = RESULTS_DIR / "final_evaluation.json"


def test_load_feature_config_matches_config_features_py_directly():
    config = load_feature_config()

    assert config["feature_columns"] == list(FEATURE_COLUMNS)
    assert config["lag_steps"] == list(LAG_STEPS) == [1, 2, 3, 4, 5, 6, 144]
    assert config["rolling_windows"] == list(ROLLING_WINDOWS) == [6, 18]
    assert config["scaled_columns"] == list(SCALED_COLUMNS)


def test_load_walk_forward_leaderboard_matches_summary_json_per_model_means():
    with open(WALK_FORWARD_SUMMARY_PATH) as f:
        raw = json.load(f)
    expected_means = raw["summary"]["per_model_means"]

    table = load_walk_forward_leaderboard()

    assert isinstance(table, pd.DataFrame)
    assert set(table["model"]) == set(expected_means.keys())
    for _, row in table.iterrows():
        expected = expected_means[row["model"]]
        assert row["mean_mae"] == expected["mean_mae"]
        assert row["mean_rmse"] == expected["mean_rmse"]
        assert row["mean_mape"] == expected["mean_mape"]

    # Spot-check specific values against the real file, not just structural
    # equality (task-quoted example: linear_regression's mean RMSE).
    lr_row = table[table["model"] == "linear_regression"].iloc[0]
    assert lr_row["mean_rmse"] == 87.39983982011466
    assert round(lr_row["mean_rmse"], 2) == 87.40


def test_load_verdicts_all_six_are_not_shown_matching_summary_json():
    with open(WALK_FORWARD_SUMMARY_PATH) as f:
        raw = json.load(f)
    expected_verdicts = raw["summary"]["verdicts"]

    verdicts = load_verdicts()

    assert len(verdicts) == 6  # 3 neural models x 2 references
    assert (verdicts["verdict"] == "not shown").all()
    for _, row in verdicts.iterrows():
        expected = expected_verdicts[row["model"]][row["reference"]]
        assert row["verdict"] == expected["verdict"]
        assert row["mean_mae_ratios"] == expected["mean_mae_ratios"]
        assert row["mean_rmse_ratios"] == expected["mean_rmse_ratios"]


def test_load_per_fold_table_fold1_naive_persistence_matches_the_single_raw_record():
    """fold=1, model=naive_persistence has exactly one record (seed=null) in
    the raw JSONL, so its grouped mean must equal that record's values
    exactly — checked against the raw file's first line directly, not via
    the same aggregation code path."""
    with open(WALK_FORWARD_RECORDS_PATH) as f:
        first_record = json.loads(f.readline())
    assert first_record["fold"] == 1
    assert first_record["model"] == "naive_persistence"

    table = load_per_fold_table()
    row = table[(table["fold"] == 1) & (table["model"] == "naive_persistence")].iloc[0]

    assert row["mae"] == first_record["mae"]
    assert row["rmse"] == first_record["rmse"]
    assert row["mape"] == first_record["mape"]


def test_load_per_fold_table_has_104_raw_records_grouped_into_expected_row_count():
    with open(WALK_FORWARD_RECORDS_PATH) as f:
        n_raw_records = sum(1 for _ in f)
    assert n_raw_records == 104  # per results/walk_forward_summary.json provenance.n_records

    table = load_per_fold_table()
    # 8 folds x 7 models (naive_persistence, naive_seasonal, linear_regression,
    # random_forest, lstm, gru, cnn_lstm) = 56 grouped (fold, model) rows.
    assert len(table) == 56


def test_load_tier1_summary_matches_final_evaluation_json():
    with open(FINAL_EVALUATION_PATH) as f:
        raw = json.load(f)

    tier1 = load_tier1_summary()

    assert tier1["shown"] == raw["tier1"]["shown"] is False
    assert tier1["threshold_mae"] == raw["tier1"]["threshold_mae"]
    assert tier1["threshold_rmse"] == raw["tier1"]["threshold_rmse"]
    assert tier1["ref_mae"] == raw["tier1_reference"]["ref_mae"] == 42.39131825471337
    assert tier1["ref_rmse"] == raw["tier1_reference"]["ref_rmse"] == 80.61873658942119

    runs = {r["seed"]: r for r in raw["horizons"]["6"]["runs"]}
    assert [r["seed"] for r in tier1["runs"]] == [42, 43, 44]
    for row in tier1["runs"]:
        assert row["test_mae"] == runs[row["seed"]]["test_mae"]
        assert row["test_rmse"] == runs[row["seed"]]["test_rmse"]
        assert row["mae_pass"] == (row["test_mae"] <= tier1["threshold_mae"])
        assert row["rmse_pass"] == (row["test_rmse"] <= tier1["threshold_rmse"])


def test_every_recorded_model_has_a_display_label():
    table = load_walk_forward_leaderboard()
    assert set(table["model"]) <= set(MODEL_LABELS)
    assert table["label"].notna().all()


def test_available_images_lists_only_files_that_actually_exist_on_disk():
    images = available_images()

    for _, path in images.items():
        assert path.exists(), f"{path} does not exist on disk"
        assert path.suffix == ".png"

    assert set(images.keys()) == {
        "walk_forward_folds",
        "walk_forward_means",
        "walk_forward_ratios",
        "final_predictions_h6",
    }


def test_bundle_schemas_cover_the_five_served_families_with_one_feature_contract():
    schemas = load_bundle_schemas()
    assert list(schemas) == ["linear_regression", "random_forest", "lstm", "gru", "cnn_lstm"]
    for family, schema in schemas.items():
        assert schema["model_family"] == family
        assert schema["feature_columns"] == list(FEATURE_COLUMNS)
        assert schema["horizon"] == 6
        assert schema["seed_end"] == "2016-04-29 23:50:00"
