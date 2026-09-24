"""Pure verdict aggregator for the registered Phase 5 walk-forward comparison
rule (DECISIONS.md, "Phase 5: walk-forward comparison rule (pre-registration)").

Inputs are per-fold ratios (model metric / reference metric) supplied by the
caller for each neural seed. No I/O, no mutation of the inputs."""

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from config.walk_forward import BETTER_MARGIN, N_FOLDS, NEURAL_SEEDS, WORSE_MARGIN


@dataclass(frozen=True)
class WalkForwardVerdict:
    verdict: str
    mean_mae_ratios: tuple[float, ...]
    mean_rmse_ratios: tuple[float, ...]


def _validate_ratios_by_seed(ratios_by_seed: Mapping[int, Sequence[float]]) -> None:
    if set(ratios_by_seed.keys()) != set(NEURAL_SEEDS):
        raise ValueError(
            f"seed keys must be exactly {NEURAL_SEEDS}, got {sorted(ratios_by_seed.keys())}"
        )
    for seed in NEURAL_SEEDS:
        values = ratios_by_seed[seed]
        if len(values) != N_FOLDS:
            raise ValueError(
                f"seed {seed} has {len(values)} ratios, expected {N_FOLDS}"
            )
        for v in values:
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"seed {seed} has a non-finite or non-positive ratio: {v}")


def mean_ratios(
    ratios_by_seed: Mapping[int, Sequence[float]],
    *,
    first_k: int | None = None,
) -> tuple[float, ...]:
    _validate_ratios_by_seed(ratios_by_seed)
    if first_k is not None and not (1 <= first_k <= N_FOLDS):
        raise ValueError(f"first_k must be between 1 and {N_FOLDS}, got {first_k}")

    means = []
    for seed in NEURAL_SEEDS:
        values = ratios_by_seed[seed]
        values = values[:first_k] if first_k is not None else values
        means.append(math.fsum(values) / len(values))
    return tuple(means)


def walk_forward_verdict(
    mae_ratios: Mapping[int, Sequence[float]],
    rmse_ratios: Mapping[int, Sequence[float]],
) -> WalkForwardVerdict:
    _validate_ratios_by_seed(mae_ratios)
    _validate_ratios_by_seed(rmse_ratios)
    if set(mae_ratios.keys()) != set(rmse_ratios.keys()):
        raise ValueError("mae_ratios and rmse_ratios must have the same seed keys")

    mean_mae = mean_ratios(mae_ratios)
    mean_rmse = mean_ratios(rmse_ratios)

    if all(m <= BETTER_MARGIN for m in mean_mae) and all(m <= BETTER_MARGIN for m in mean_rmse):
        verdict = "shown better"
    elif all(m >= WORSE_MARGIN for m in mean_mae) and all(m >= WORSE_MARGIN for m in mean_rmse):
        verdict = "shown worse"
    else:
        verdict = "not shown"

    return WalkForwardVerdict(
        verdict=verdict, mean_mae_ratios=mean_mae, mean_rmse_ratios=mean_rmse
    )
