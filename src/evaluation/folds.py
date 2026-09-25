"""Pure expanding-window fold generator for the registered Phase 5 walk-forward
scheme (decisions.md, ADR-009).

No I/O, no mutation of the input, and no label purge — the harness applies
mask_ineligible_labels separately."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FoldSpec:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp
    n_train: int
    n_eval: int


def make_folds(
    timestamps: pd.DatetimeIndex,
    *,
    first_eval_start,
    n_folds: int,
    eval_days: int,
    freq_minutes: int,
    holdout_start,
) -> tuple[FoldSpec, ...]:
    if len(timestamps) == 0:
        raise ValueError("timestamps must not be empty")
    if not timestamps.is_monotonic_increasing or timestamps.has_duplicates:
        raise ValueError("timestamps must be strictly increasing")

    first_eval_start = pd.Timestamp(first_eval_start)
    holdout_start = pd.Timestamp(holdout_start)
    train_start = timestamps[0]
    expected_n_eval = eval_days * 24 * 60 // freq_minutes

    folds = []
    for k in range(1, n_folds + 1):
        eval_start = first_eval_start + pd.Timedelta(days=eval_days * (k - 1))
        eval_end = eval_start + pd.Timedelta(days=eval_days)

        if eval_end > holdout_start:
            raise ValueError(
                f"fold {k} evaluation window ends at {eval_end}, "
                f"past holdout_start {holdout_start}"
            )

        n_train = int(timestamps.searchsorted(eval_start, side="left"))
        if n_train == 0:
            raise ValueError(f"fold {k} has no training rows before {eval_start}")

        n_eval = int(
            timestamps.searchsorted(eval_end, side="left")
            - timestamps.searchsorted(eval_start, side="left")
        )
        if n_eval != expected_n_eval:
            raise ValueError(
                f"fold {k} has {n_eval} evaluation rows, expected {expected_n_eval}"
            )

        folds.append(
            FoldSpec(
                fold=k,
                train_start=train_start,
                train_end=eval_start,
                eval_start=eval_start,
                eval_end=eval_end,
                n_train=n_train,
                n_eval=n_eval,
            )
        )

    return tuple(folds)
