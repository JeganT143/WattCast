"""Golden baseline fixture: compute, and optionally write, all baseline metrics.

Run from the repo root:  python -m tests.golden --out PATH
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from config.features import MAPE_THRESHOLD
from src.evaluation.harness import evaluate_forecaster
from src.models.naive import NaivePersistenceForecaster, NaiveSeasonalForecaster
from src.models.sklearn_models import LinearRegressionForecaster, RandomForestForecaster

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "golden_baselines.json"
PROCESSED = PROJECT_ROOT / "data" / "processed"
HORIZONS = (1, 6)
PARTITIONS = ("train", "val", "test")
METRICS = ("mae", "rmse", "mape")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, cwd=PROJECT_ROOT).strip()


def _meta() -> dict:
    return {
        "git_commit_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
    }


def _load(split: str, h: int) -> pd.DataFrame:
    return pd.read_csv(PROCESSED / f"{split}_t{h}.csv", parse_dates=["date"])


def _models(h: int) -> dict:
    return {
        "naive_persistence": NaivePersistenceForecaster(),
        "naive_seasonal": NaiveSeasonalForecaster(h),
        "linear_regression": LinearRegressionForecaster(),
        "random_forest": RandomForestForecaster(),
    }


def compute() -> dict:
    results = {}
    for h in HORIZONS:
        frames = {p: _load(p, h) for p in PARTITIONS}
        for name, forecaster in _models(h).items():
            if name == "random_forest":
                assert forecaster.params.get("random_state") is not None, (
                    "RandomForestForecaster has no explicit random_state; "
                    "fix and log that before creating a golden fixture"
                )
            result = evaluate_forecaster(
                forecaster,
                train_df=frames["train"],
                val_df=frames["val"],
                test_df=frames["test"],
                target_column=f"target_t{h}",
                model_name=name,
                horizon=h,
                mape_threshold=MAPE_THRESHOLD,
            )
            results[f"{name}|h{h}"] = {
                "metrics": {
                    f"{p}_{m}": float(result.metrics[p][m])
                    for p in PARTITIONS
                    for m in METRICS
                },
                "n_rows": {p: len(frames[p]) for p in PARTITIONS},
                "n_evaluated": {p: int(len(result.predictions[p])) for p in PARTITIONS},
            }
    return {"meta": _meta(), "results": results}


def dumps(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dumps(compute()))
    print(f"wrote {args.out}")
