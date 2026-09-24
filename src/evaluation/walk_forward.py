"""Walk-forward runner — an orchestrator that adds no evaluation logic of its
own. It builds each fold's data once (build_fold_datasets), fits a fresh
forecaster per (fold, model, seed) via the supplied factories, and scores it
with evaluate_on_validation. No scoring, context, purge or scaling logic
lives here."""

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import pandas as pd

from config.features import MAPE_THRESHOLD
from config.walk_forward import NEURAL_SEEDS
from src.evaluation import metrics
from src.evaluation.fold_data import build_fold_datasets
from src.evaluation.folds import FoldSpec
from src.evaluation.validation_only import evaluate_on_validation
from src.models.forecaster import Forecaster


@dataclass(frozen=True)
class FoldRunRecord:
    fold: int
    model: str
    seed: int | None
    mae: float
    rmse: float
    mape: float
    n_evaluated: int
    n_warmup_rows: int
    n_ineligible_label_rows: int


def run_walk_forward(
    df_features: pd.DataFrame,
    specs: Sequence[FoldSpec],
    *,
    horizon: int,
    deterministic: Mapping[str, Callable[[], Forecaster]],
    seeded: Mapping[str, Callable[[int], Forecaster]],
    seeds: Sequence[int] = NEURAL_SEEDS,
    on_record: Callable[[FoldRunRecord], None] | None = None,
) -> tuple[FoldRunRecord, ...]:
    if len(specs) == 0:
        raise ValueError("specs must not be empty")
    if len(seeds) == 0:
        raise ValueError("seeds must not be empty")
    if len(deterministic) == 0 and len(seeded) == 0:
        raise ValueError("deterministic and seeded must not both be empty")
    overlap = set(deterministic) & set(seeded)
    if overlap:
        raise ValueError(f"model names appear in both mappings: {sorted(overlap)}")

    target_column = f"target_t{horizon}"
    records: list[FoldRunRecord] = []

    for spec in specs:
        fd = build_fold_datasets(df_features, spec, horizon)

        for name, factory in deterministic.items():
            forecaster = factory()
            result = evaluate_on_validation(forecaster, fd.train_df, fd.eval_df, target_column)
            record = FoldRunRecord(
                fold=spec.fold,
                model=name,
                seed=None,
                mae=result.metrics["mae"],
                rmse=result.metrics["rmse"],
                mape=metrics.mape(result.actuals, result.predictions, MAPE_THRESHOLD),
                n_evaluated=result.n_evaluated,
                n_warmup_rows=result.n_warmup_rows,
                n_ineligible_label_rows=result.n_ineligible_label_rows,
            )
            records.append(record)
            if on_record is not None:
                on_record(record)

        for name, factory in seeded.items():
            for seed in seeds:
                forecaster = factory(seed)
                result = evaluate_on_validation(forecaster, fd.train_df, fd.eval_df, target_column)
                record = FoldRunRecord(
                    fold=spec.fold,
                    model=name,
                    seed=seed,
                    mae=result.metrics["mae"],
                    rmse=result.metrics["rmse"],
                    mape=metrics.mape(result.actuals, result.predictions, MAPE_THRESHOLD),
                    n_evaluated=result.n_evaluated,
                    n_warmup_rows=result.n_warmup_rows,
                    n_ineligible_label_rows=result.n_ineligible_label_rows,
                )
                records.append(record)
                if on_record is not None:
                    on_record(record)

    return tuple(records)
