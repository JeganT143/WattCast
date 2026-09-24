"""Materializes run_pipeline() output as the six partition CSVs and writes nothing else
(scaler_train_fit.joblib and split_boundaries.json are deliberately untouched).

Usage:
    python -m src.build_processed --out DIR
"""

import argparse
from pathlib import Path

import pandas as pd

from config.paths import RAW_DATA_PATH
from src.pipeline import PipelineResult, run_pipeline

_FRAMES = ("train_t1", "val_t1", "test_t1", "train_t6", "val_t6", "test_t6")


def write_partitions(result: PipelineResult, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in _FRAMES:
        path = out_dir / f"{name}.csv"
        getattr(result, name).to_csv(path, index=False)
        written.append(path)
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    df_raw = pd.read_csv(RAW_DATA_PATH, parse_dates=["date"])
    result = run_pipeline(df_raw)
    paths = write_partitions(result, args.out)
    for path in paths:
        print(f"wrote {path}")
