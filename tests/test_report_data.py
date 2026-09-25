"""Tests for src/ui/report_data.py: every function must read a REAL source
file (results/*.json, results/*.jsonl, DECISIONS.md, README.md,
config/features.py) — never a mock, never a hardcoded literal that wasn't
independently cross-checked against that file. Where practical, a test
re-reads the same source file itself (independent of the function under
test) and asserts the function's output matches, rather than encoding a
number that was merely copy-pasted once.
"""

import json
import re

import pandas as pd

from config.features import FEATURE_COLUMNS, LAG_STEPS, ROLLING_WINDOWS, SCALED_COLUMNS
from config.paths import PROJECT_ROOT
from src.ui.report_data import (
    available_images,
    load_dataset_summary,
    load_eda_summary,
    load_feature_config,
    load_limitations,
    load_per_fold_table,
    load_tier1_summary,
    load_verdicts,
    load_walk_forward_leaderboard,
)

DECISIONS_PATH = PROJECT_ROOT / "DECISIONS.md"
README_PATH = PROJECT_ROOT / "README.md"
WALK_FORWARD_SUMMARY_PATH = PROJECT_ROOT / "results" / "walk_forward_summary.json"
WALK_FORWARD_RECORDS_PATH = PROJECT_ROOT / "results" / "walk_forward_records.jsonl"
FINAL_EVALUATION_PATH = PROJECT_ROOT / "results" / "final_evaluation.json"


def test_load_dataset_summary_matches_readme_sentence_independently_reparsed():
    text = README_PATH.read_text()
    match = re.search(
        r"([\d,]+) rows, (\d+) columns, 10-minute sensor intervals spanning "
        r"~([\d.]+) months \((\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})\)",
        text,
    )
    assert match is not None, "expected dataset sentence not found in README.md"
    expected_rows, expected_cols, expected_months, expected_start, expected_end = match.groups()

    summary = load_dataset_summary()

    assert summary["n_rows"] == int(expected_rows.replace(",", "")) == 19735
    assert summary["n_columns"] == int(expected_cols) == 29
    assert summary["span_months"] == float(expected_months) == 4.5
    assert summary["start_date"] == expected_start == "2016-01-11"
    assert summary["end_date"] == expected_end == "2016-05-27"
    assert summary["interval_minutes"] == 10


def test_load_eda_summary_bullets_are_verbatim_from_decisions_md():
    text = DECISIONS_PATH.read_text()
    lines = text.splitlines()
    heading_idx = lines.index("## Phase 1 — EDA Key Findings")
    expected_bullets = []
    for line in lines[heading_idx + 1 :]:
        if line.strip().startswith("## "):
            break
        if line.strip().startswith("- "):
            expected_bullets.append(line.strip()[2:])

    bullets = load_eda_summary()

    assert bullets == expected_bullets
    assert len(bullets) > 0
    assert any("19,735 rows" in b for b in bullets)
    assert any("ADF" in b for b in bullets)


def test_load_feature_config_matches_config_features_py_directly():
    config = load_feature_config()

    assert config["feature_columns"] == list(FEATURE_COLUMNS)
    assert config["lag_steps"] == list(LAG_STEPS) == [1, 2, 3, 4, 5, 6, 144]
    assert config["rolling_windows"] == list(ROLLING_WINDOWS) == [6, 18]
    assert config["scaled_columns"] == list(SCALED_COLUMNS)


def test_load_limitations_bullets_are_verbatim_from_decisions_md():
    text = DECISIONS_PATH.read_text()
    lines = text.splitlines()
    heading_idx = lines.index("### Known limits (registered, restated)")
    expected_bullets = []
    for line in lines[heading_idx + 1 :]:
        if line.strip().startswith("#"):
            break
        if line.strip().startswith("- "):
            expected_bullets.append(line.strip()[2:])

    limitations = load_limitations()

    assert limitations == expected_bullets
    assert len(limitations) > 0
    assert any("138-day" in item for item in limitations)
    assert any("one-sided" in item for item in limitations)


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
