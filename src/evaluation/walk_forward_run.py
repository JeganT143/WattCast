"""Orchestrates a walk-forward run: wraps run_walk_forward with an exclusive-create JSONL
writer (one flushed, fsynced line per record, never overwritten) and applies the
registered summary once, with provenance. No scoring, evaluation or training logic of
its own — deterministic_factories/seeded_factories supply the models and
summarize_walk_forward supplies the verdicts.
"""

import argparse
import dataclasses
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from config.features import SPLIT_VAL_END, TARGET_HORIZONS
from config.paths import RAW_DATA_PATH
from config.walk_forward import EVAL_DAYS, FIRST_EVAL_START, FREQ_MINUTES, N_FOLDS, NEURAL_SEEDS
from src.evaluation.folds import FoldSpec, make_folds
from src.evaluation.walk_forward import FoldRunRecord, run_walk_forward
from src.evaluation.walk_forward_models import deterministic_factories, seeded_factories
from src.evaluation.walk_forward_summary import summarize_walk_forward
from src.features.build_features import build_features
from src.models.forecaster import Forecaster
from src.pipeline import run_pipeline

RECORDS_NAME = "walk_forward_records.jsonl"
SUMMARY_NAME = "walk_forward_summary.json"


def execute(
    df_features: pd.DataFrame,
    specs: Sequence[FoldSpec],
    *,
    horizon: int,
    deterministic: Mapping[str, Callable[[], Forecaster]],
    seeded: Mapping[str, Callable[[int], Forecaster]],
    records_path,
    on_progress: Callable[[FoldRunRecord], None] | None = None,
) -> tuple[FoldRunRecord, ...]:
    records_path = Path(records_path)
    if records_path.exists():
        raise FileExistsError(str(records_path))
    records_path.parent.mkdir(parents=True, exist_ok=True)

    with open(records_path, "x") as f:

        def cb(record: FoldRunRecord) -> None:
            f.write(json.dumps(dataclasses.asdict(record), sort_keys=True, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            if on_progress is not None:
                on_progress(record)

        records = run_walk_forward(
            df_features,
            specs,
            horizon=horizon,
            deterministic=deterministic,
            seeded=seeded,
            on_record=cb,
        )

    return records


def write_summary(
    records: Sequence[FoldRunRecord],
    summary_path,
    *,
    neural_models: Sequence[str],
    provenance: dict,
) -> dict:
    summary_path = Path(summary_path)
    if summary_path.exists():
        raise FileExistsError(str(summary_path))

    result = {
        "provenance": provenance,
        "summary": summarize_walk_forward(records, neural_models=neural_models),
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "x") as f:
        f.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")

    return result


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _git_dirty() -> bool:
    return bool(subprocess.check_output(["git", "status", "--short"], text=True).strip())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--deterministic-only", action="store_true")
    parser.add_argument("--horizon", type=int, default=6)
    args = parser.parse_args(argv)

    if not (1 <= args.folds <= N_FOLDS):
        raise ValueError(f"--folds must be between 1 and {N_FOLDS}, got {args.folds}")

    out_dir = Path(args.out_dir)
    records_path = out_dir / RECORDS_NAME
    summary_path = out_dir / SUMMARY_NAME
    full_run = args.folds == N_FOLDS and not args.deterministic_only

    if records_path.exists():
        raise FileExistsError(str(records_path))
    if full_run and summary_path.exists():
        raise FileExistsError(str(summary_path))

    start_time = time.perf_counter()
    start_iso = datetime.now(timezone.utc).isoformat()
    git_commit_sha = _git_head()
    git_dirty = _git_dirty()
    seeds = list(NEURAL_SEEDS)

    df_raw = pd.read_csv(RAW_DATA_PATH, parse_dates=["date"])
    df_features = build_features(df_raw, target_horizons=TARGET_HORIZONS)
    result = run_pipeline(df_raw)
    timestamps = pd.DatetimeIndex(
        pd.concat(
            [result.train_t6["date"], result.val_t6["date"], result.test_t6["date"]],
            ignore_index=True,
        )
    )
    specs = make_folds(
        timestamps,
        first_eval_start=pd.Timestamp(FIRST_EVAL_START),
        n_folds=N_FOLDS,
        eval_days=EVAL_DAYS,
        freq_minutes=FREQ_MINUTES,
        holdout_start=pd.Timestamp(SPLIT_VAL_END),
    )[: args.folds]

    deterministic = deterministic_factories(args.horizon)
    seeded = {} if args.deterministic_only else seeded_factories()

    fold_table = [
        {
            "fold": spec.fold,
            "train_start": str(spec.train_start),
            "train_end": str(spec.train_end),
            "eval_start": str(spec.eval_start),
            "eval_end": str(spec.eval_end),
            "n_train": int(spec.n_train),
            "n_eval": int(spec.n_eval),
        }
        for spec in specs
    ]

    provenance = {
        "git_commit_sha": git_commit_sha,
        "git_dirty": git_dirty,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "horizon": args.horizon,
        "seeds": seeds,
        "folds": args.folds,
        "fold_table": fold_table,
        "models": list(deterministic) + list(seeded),
        "start_time_utc": start_iso,
    }

    count = 0

    def on_progress(record: FoldRunRecord) -> None:
        nonlocal count
        count += 1
        elapsed = time.perf_counter() - start_time
        print(
            f"[{count}] fold {record.fold} model {record.model} seed {record.seed} "
            f"mae={record.mae} rmse={record.rmse} elapsed={elapsed:.1f}s",
            flush=True,
        )

    records = execute(
        df_features,
        specs,
        horizon=args.horizon,
        deterministic=deterministic,
        seeded=seeded,
        records_path=records_path,
        on_progress=on_progress,
    )

    if full_run:
        write_summary(records, summary_path, neural_models=list(seeded), provenance=provenance)
        print(summary_path)

    print(f"wrote {records_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
