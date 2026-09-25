"""Loaders for the result files the Streamlit app displays.

Every number shown in the app's result tables is read here from a recorded
file in results/ (or derived from one by a plain mean), never retyped:

- results/walk_forward_summary.json   8-fold walk-forward means and verdicts
- results/walk_forward_records.jsonl  one record per (fold, model, seed)
- results/final_evaluation.json       the one-shot held-out test check
- models/<family>/schema.json         the contract of each deployed model
"""

import json

import pandas as pd

from config.features import FEATURE_COLUMNS, LAG_STEPS, ROLLING_WINDOWS, SCALED_COLUMNS
from config.paths import MODELS_DIR, PROJECT_ROOT, RESULTS_DIR

WALK_FORWARD_SUMMARY_PATH = RESULTS_DIR / "walk_forward_summary.json"
WALK_FORWARD_RECORDS_PATH = RESULTS_DIR / "walk_forward_records.jsonl"
FINAL_EVALUATION_PATH = RESULTS_DIR / "final_evaluation.json"
DOCS_IMG_DIR = PROJECT_ROOT / "docs" / "img"

MODEL_LABELS = {
    "naive_persistence": "Naive persistence",
    "naive_seasonal": "Naive seasonal",
    "linear_regression": "Linear regression",
    "random_forest": "Random forest",
    "lstm": "LSTM",
    "gru": "GRU",
    "cnn_lstm": "CNN-LSTM",
}

MODEL_GROUPS = {
    "naive_persistence": "Naive baseline",
    "naive_seasonal": "Naive baseline",
    "linear_regression": "Classical",
    "random_forest": "Classical",
    "lstm": "Deep sequence",
    "gru": "Deep sequence",
    "cnn_lstm": "Deep sequence",
}


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def _read_json(path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_feature_config() -> dict:
    """The feature contract, imported from config/features.py."""
    return {
        "feature_columns": list(FEATURE_COLUMNS),
        "lag_steps": list(LAG_STEPS),
        "rolling_windows": list(ROLLING_WINDOWS),
        "scaled_columns": list(SCALED_COLUMNS),
    }


def load_walk_forward_leaderboard() -> pd.DataFrame:
    """Mean MAE/RMSE/MAPE over the 8 folds (and over 3 seeds for the deep
    models), sorted by mean MAE."""
    means = _read_json(WALK_FORWARD_SUMMARY_PATH)["summary"]["per_model_means"]
    rows = [
        {
            "model": model,
            "label": model_label(model),
            "group": MODEL_GROUPS.get(model, ""),
            "mean_mae": vals["mean_mae"],
            "mean_rmse": vals["mean_rmse"],
            "mean_mape": vals["mean_mape"],
        }
        for model, vals in means.items()
    ]
    return pd.DataFrame(rows).sort_values("mean_mae").reset_index(drop=True)


def load_verdicts() -> pd.DataFrame:
    """The six registered (deep model, reference) walk-forward verdicts with
    their per-seed mean MAE and RMSE ratios (seeds 42, 43, 44)."""
    verdicts = _read_json(WALK_FORWARD_SUMMARY_PATH)["summary"]["verdicts"]
    rows = [
        {
            "model": model,
            "reference": reference,
            "verdict": v["verdict"],
            "mean_mae_ratios": v["mean_mae_ratios"],
            "mean_rmse_ratios": v["mean_rmse_ratios"],
        }
        for model, by_reference in verdicts.items()
        for reference, v in by_reference.items()
    ]
    return pd.DataFrame(rows)


def load_per_fold_table() -> pd.DataFrame:
    """Per-(fold, model) metrics, averaged over seeds where a model has
    several (the deep models have 3 seeded records per fold)."""
    with open(WALK_FORWARD_RECORDS_PATH) as f:
        records = [json.loads(line) for line in f]
    return (
        pd.DataFrame(records)
        .groupby(["fold", "model"], as_index=False)[["mae", "rmse", "mape"]]
        .mean()
        .sort_values(["fold", "model"])
        .reset_index(drop=True)
    )


def load_tier1_summary() -> dict:
    """The one-shot held-out test check at h=6: LSTM (3 seeds) against
    0.99 x the linear-regression test MAE and RMSE."""
    data = _read_json(FINAL_EVALUATION_PATH)
    passes = {p["seed"]: p for p in data["tier1"]["per_seed"]}
    runs = [
        {
            "seed": run["seed"],
            "test_mae": run["test_mae"],
            "test_rmse": run["test_rmse"],
            "mae_pass": passes[run["seed"]]["mae_pass"],
            "rmse_pass": passes[run["seed"]]["rmse_pass"],
        }
        for run in data["horizons"]["6"]["runs"]
    ]
    return {
        "shown": data["tier1"]["shown"],
        "threshold_mae": data["tier1"]["threshold_mae"],
        "threshold_rmse": data["tier1"]["threshold_rmse"],
        "ref_mae": data["tier1_reference"]["ref_mae"],
        "ref_rmse": data["tier1_reference"]["ref_rmse"],
        "per_seed": data["tier1"]["per_seed"],
        "runs": runs,
    }


def load_bundle_schemas() -> dict[str, dict]:
    """schema.json of every deployed bundle, keyed by family in display
    order. Reads JSON only — no model is loaded."""
    schemas = {}
    for family in MODEL_LABELS:
        path = MODELS_DIR / family / "schema.json"
        if path.exists():
            schemas[family] = _read_json(path)
    return schemas


def available_images() -> dict:
    return {
        "walk_forward_folds": DOCS_IMG_DIR / "walk_forward_folds.png",
        "walk_forward_means": DOCS_IMG_DIR / "walk_forward_means.png",
        "walk_forward_ratios": DOCS_IMG_DIR / "walk_forward_ratios.png",
        "final_predictions_h6": RESULTS_DIR / "final_predictions_h6.png",
    }
