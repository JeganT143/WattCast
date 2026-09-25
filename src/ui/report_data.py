"""Pure, pytest-tested functions that load Streamlit report content
directly from already-registered source files — results/*.json,
results/walk_forward_records.jsonl, DECISIONS.md, README.md, and
config/features.py. Each function reads (or does a direct arithmetic
aggregation of) a real file and cites its source in its own docstring.
Nothing here invents a number: a value not present in a source file is
either read as-is or, for the per-fold table, computed by a plain mean over
raw recorded rows — never estimated or guessed.
"""

import json
import re

import pandas as pd

from config.features import FEATURE_COLUMNS, LAG_STEPS, ROLLING_WINDOWS, SCALED_COLUMNS
from config.paths import PROJECT_ROOT

DECISIONS_PATH = PROJECT_ROOT / "DECISIONS.md"
README_PATH = PROJECT_ROOT / "README.md"
WALK_FORWARD_SUMMARY_PATH = PROJECT_ROOT / "results" / "walk_forward_summary.json"
WALK_FORWARD_RECORDS_PATH = PROJECT_ROOT / "results" / "walk_forward_records.jsonl"
FINAL_EVALUATION_PATH = PROJECT_ROOT / "results" / "final_evaluation.json"
DOCS_IMG_DIR = PROJECT_ROOT / "docs" / "img"
RESULTS_DIR = PROJECT_ROOT / "results"


def _bullets_after_heading(lines: list[str], heading: str) -> list[str]:
    heading_idx = None
    for i, line in enumerate(lines):
        if line.strip() == heading:
            heading_idx = i
            break
    if heading_idx is None:
        raise ValueError(f"heading {heading!r} not found in DECISIONS.md")

    bullets = []
    for line in lines[heading_idx + 1 :]:
        stripped = line.strip()
        if stripped.startswith("#"):
            break
        if stripped.startswith("- "):
            bullets.append(stripped[2:])
    if not bullets:
        raise ValueError(f"no bullet items found under heading {heading!r}")
    return bullets


def load_dataset_summary() -> dict:
    """Source: README.md's opening description sentence — "<rows> rows,
    <cols> columns, 10-minute sensor intervals spanning ~<months> months
    (<start> to <end>)"."""
    text = README_PATH.read_text()
    match = re.search(
        r"([\d,]+) rows, (\d+) columns, 10-minute sensor intervals spanning "
        r"~([\d.]+) months \((\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})\)",
        text,
    )
    if match is None:
        raise ValueError("could not find the dataset summary sentence in README.md")
    n_rows, n_columns, months, start, end = match.groups()
    return {
        "n_rows": int(n_rows.replace(",", "")),
        "n_columns": int(n_columns),
        "interval_minutes": 10,
        "span_months": float(months),
        "start_date": start,
        "end_date": end,
        "source": "README.md",
    }


def load_eda_summary() -> list[str]:
    """Source: DECISIONS.md, "## Phase 1 — EDA Key Findings" bullet list."""
    lines = DECISIONS_PATH.read_text().splitlines()
    return _bullets_after_heading(lines, "## Phase 1 — EDA Key Findings")


def load_feature_config() -> dict:
    """Source: config/features.py (FEATURE_COLUMNS, LAG_STEPS,
    ROLLING_WINDOWS, SCALED_COLUMNS) — imported directly, never re-parsed
    or retyped."""
    return {
        "feature_columns": list(FEATURE_COLUMNS),
        "lag_steps": list(LAG_STEPS),
        "rolling_windows": list(ROLLING_WINDOWS),
        "scaled_columns": list(SCALED_COLUMNS),
    }


def load_limitations() -> list[str]:
    """Source: DECISIONS.md, "### Known limits (registered, restated)"
    bullet list (Phase 5 walk-forward run-results section)."""
    lines = DECISIONS_PATH.read_text().splitlines()
    return _bullets_after_heading(lines, "### Known limits (registered, restated)")


def load_walk_forward_leaderboard() -> pd.DataFrame:
    """Source: results/walk_forward_summary.json,
    summary.per_model_means — the registered mean MAE/RMSE/MAPE over 8
    folds (and over 3 seeds for the neural models), read as-is."""
    with open(WALK_FORWARD_SUMMARY_PATH) as f:
        summary = json.load(f)
    means = summary["summary"]["per_model_means"]
    rows = [
        {
            "model": model,
            "mean_mae": vals["mean_mae"],
            "mean_rmse": vals["mean_rmse"],
            "mean_mape": vals["mean_mape"],
        }
        for model, vals in means.items()
    ]
    return pd.DataFrame(rows).sort_values("mean_mae").reset_index(drop=True)


def load_verdicts() -> pd.DataFrame:
    """Source: results/walk_forward_summary.json, summary.verdicts — the
    six registered (neural model, reference) walk-forward verdicts (all
    "not shown" per the pre-registered rule, DECISIONS.md "Phase 5:
    walk-forward comparison rule")."""
    with open(WALK_FORWARD_SUMMARY_PATH) as f:
        summary = json.load(f)
    verdicts = summary["summary"]["verdicts"]
    rows = []
    for model, by_ref in verdicts.items():
        for reference, v in by_ref.items():
            rows.append(
                {
                    "model": model,
                    "reference": reference,
                    "verdict": v["verdict"],
                    "mean_mae_ratios": v["mean_mae_ratios"],
                    "mean_rmse_ratios": v["mean_rmse_ratios"],
                }
            )
    return pd.DataFrame(rows)


def load_per_fold_table() -> pd.DataFrame:
    """Source: results/walk_forward_records.jsonl — one JSON line per
    (fold, model, seed) record. Grouped by (fold, model) and averaged over
    seeds (naive/linear_regression/random_forest have a single seed=null
    record per fold; the three neural models have 3 seeded records per
    fold) — a direct arithmetic aggregate of the raw recorded rows, not a
    separate or re-derived computation."""
    records = []
    with open(WALK_FORWARD_RECORDS_PATH) as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    return (
        df.groupby(["fold", "model"], as_index=False)[["mae", "rmse", "mape"]]
        .mean()
        .sort_values(["fold", "model"])
        .reset_index(drop=True)
    )


def load_tier1_summary() -> dict:
    """Source: results/final_evaluation.json's "tier1"/"tier1_reference"
    sections — the Phase 4 single-window LSTM-vs-linear_regression (h=6)
    Tier 1 verdict."""
    with open(FINAL_EVALUATION_PATH) as f:
        data = json.load(f)
    return {
        "shown": data["tier1"]["shown"],
        "threshold_mae": data["tier1"]["threshold_mae"],
        "threshold_rmse": data["tier1"]["threshold_rmse"],
        "ref_mae": data["tier1_reference"]["ref_mae"],
        "ref_rmse": data["tier1_reference"]["ref_rmse"],
        "per_seed": data["tier1"]["per_seed"],
    }


def available_images() -> dict:
    """Source: a filesystem listing of docs/img/*.png and results/*.png —
    confirms which images actually exist, rather than assuming filenames
    from a prior stage's naming convention."""
    return {
        "walk_forward_folds": DOCS_IMG_DIR / "walk_forward_folds.png",
        "walk_forward_means": DOCS_IMG_DIR / "walk_forward_means.png",
        "walk_forward_ratios": DOCS_IMG_DIR / "walk_forward_ratios.png",
        "final_predictions_h6": RESULTS_DIR / "final_predictions_h6.png",
    }
