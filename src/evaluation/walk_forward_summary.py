"""Pure summary of walk-forward runner records into the registered
per-(neural model, reference) verdicts and descriptive Tier 2 numbers
(DECISIONS.md, "Phase 5: walk-forward comparison rule (pre-registration)").

Ratios and verdicts are computed from FoldRunRecord instances only; no
scoring, evaluation or training happens here, and walk_forward_verdict
owns the verdict rule itself. No I/O, no mutation of the input."""

import math
import statistics
from typing import Sequence

from config.walk_forward import N_FOLDS, NEURAL_SEEDS
from src.evaluation.walk_forward import FoldRunRecord
from src.evaluation.walk_forward_verdict import mean_ratios, walk_forward_verdict

REFERENCES = ("linear_regression", "random_forest")


def _validate_group_disjointness(neural_models, baseline_models, references) -> None:
    groups = {
        "neural_models": set(neural_models),
        "baseline_models": set(baseline_models),
        "references": set(references),
    }
    names = list(groups.items())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = names[i][1] & names[j][1]
            if overlap:
                raise ValueError(
                    f"names {sorted(overlap)} appear in both {names[i][0]} and {names[j][0]}"
                )


def _index_records(records: Sequence[FoldRunRecord]) -> dict:
    index: dict[tuple[int, str, int | None], FoldRunRecord] = {}
    for record in records:
        key = (record.fold, record.model, record.seed)
        if key in index:
            raise ValueError(f"duplicate (fold, model, seed) triple: {key}")
        index[key] = record
    return index


def _validate_finite(records: Sequence[FoldRunRecord]) -> None:
    for record in records:
        for field_name in ("mae", "rmse", "mape"):
            value = getattr(record, field_name)
            if not math.isfinite(value):
                raise ValueError(
                    f"record (fold={record.fold}, model={record.model}, seed={record.seed}) "
                    f"has non-finite {field_name}: {value}"
                )


def _records_for_model(index: dict, model: str, seed: int | None = "__any__") -> list:
    return [
        record
        for (fold, m, s), record in index.items()
        if m == model and (seed == "__any__" or s == seed)
    ]


def _validate_non_seeded_model(index: dict, model: str) -> None:
    records = _records_for_model(index, model)
    if not records:
        raise ValueError(f"model {model!r} is missing from the records")
    for record in records:
        if record.seed is not None:
            raise ValueError(
                f"model {model!r} must have seed None, got seed {record.seed} at fold {record.fold}"
            )
    folds = {record.fold for record in records}
    if folds != set(range(1, N_FOLDS + 1)):
        raise ValueError(
            f"model {model!r} folds must be exactly 1..{N_FOLDS}, got {sorted(folds)}"
        )


def _validate_neural_model(index: dict, model: str) -> None:
    records = _records_for_model(index, model)
    if not records:
        raise ValueError(f"model {model!r} is missing from the records")
    folds = {record.fold for record in records}
    if folds != set(range(1, N_FOLDS + 1)):
        raise ValueError(
            f"model {model!r} folds must be exactly 1..{N_FOLDS}, got {sorted(folds)}"
        )
    for fold in range(1, N_FOLDS + 1):
        seeds = {record.seed for record in records if record.fold == fold}
        if seeds != set(NEURAL_SEEDS):
            raise ValueError(
                f"model {model!r} fold {fold} seeds must be exactly {NEURAL_SEEDS}, got {sorted(s for s in seeds if s is not None)}"
            )


def _validate_reference_positive(index: dict, model: str) -> None:
    for record in _records_for_model(index, model):
        if record.mae <= 0 or record.rmse <= 0:
            raise ValueError(
                f"reference {model!r} fold {record.fold} has non-positive mae/rmse: "
                f"mae={record.mae}, rmse={record.rmse}"
            )


def _ratios_by_seed(index: dict, neural_model: str, reference: str, metric: str) -> dict:
    ratios: dict[int, list] = {}
    for seed in NEURAL_SEEDS:
        values = []
        for fold in range(1, N_FOLDS + 1):
            neural_value = getattr(index[(fold, neural_model, seed)], metric)
            reference_value = getattr(index[(fold, reference, None)], metric)
            values.append(neural_value / reference_value)
        ratios[seed] = values
    return ratios


def _mean_over_folds(index: dict, model: str, field_name: str) -> float:
    values = [getattr(record, field_name) for record in _records_for_model(index, model)]
    return math.fsum(values) / len(values)


def summarize_walk_forward(
    records: Sequence[FoldRunRecord],
    *,
    neural_models: Sequence[str],
    baseline_models: Sequence[str] = ("naive_persistence", "naive_seasonal"),
    references: Sequence[str] = REFERENCES,
) -> dict:
    if len(records) == 0:
        raise ValueError("records must not be empty")

    _validate_group_disjointness(neural_models, baseline_models, references)

    index = _index_records(records)
    _validate_finite(records)

    for model in references:
        _validate_non_seeded_model(index, model)
        _validate_reference_positive(index, model)
    for model in baseline_models:
        _validate_non_seeded_model(index, model)
    for model in neural_models:
        _validate_neural_model(index, model)

    verdicts: dict[str, dict] = {}
    for neural_model in neural_models:
        verdicts[neural_model] = {}
        for reference in references:
            mae_ratios = _ratios_by_seed(index, neural_model, reference, "mae")
            rmse_ratios = _ratios_by_seed(index, neural_model, reference, "rmse")

            result = walk_forward_verdict(mae_ratios, rmse_ratios)
            sensitivity_mae = mean_ratios(mae_ratios, first_k=6)
            sensitivity_rmse = mean_ratios(rmse_ratios, first_k=6)

            worst_fold_mae = [max(mae_ratios[seed]) for seed in NEURAL_SEEDS]
            worst_fold_rmse = [max(rmse_ratios[seed]) for seed in NEURAL_SEEDS]
            folds_won_mae = [
                sum(1 for v in mae_ratios[seed] if v < 1.0) for seed in NEURAL_SEEDS
            ]
            folds_won_rmse = [
                sum(1 for v in rmse_ratios[seed] if v < 1.0) for seed in NEURAL_SEEDS
            ]

            verdicts[neural_model][reference] = {
                "verdict": result.verdict,
                "mean_mae_ratios": list(result.mean_mae_ratios),
                "mean_rmse_ratios": list(result.mean_rmse_ratios),
                "sensitivity_folds_1_to_6": {
                    "mean_mae_ratios": list(sensitivity_mae),
                    "mean_rmse_ratios": list(sensitivity_rmse),
                },
                "worst_fold_mae_ratio": worst_fold_mae,
                "worst_fold_rmse_ratio": worst_fold_rmse,
                "folds_won_mae": folds_won_mae,
                "folds_won_rmse": folds_won_rmse,
                "seed_spread_mae": statistics.stdev(list(result.mean_mae_ratios)),
                "seed_spread_rmse": statistics.stdev(list(result.mean_rmse_ratios)),
            }

    per_model_means: dict[str, dict] = {}
    for model in (*neural_models, *references, *baseline_models):
        per_model_means[model] = {
            "mean_mae": _mean_over_folds(index, model, "mae"),
            "mean_rmse": _mean_over_folds(index, model, "rmse"),
            "mean_mape": _mean_over_folds(index, model, "mape"),
        }

    return {
        "verdicts": verdicts,
        "per_model_means": per_model_means,
        "n_records": len(records),
    }
