"""
Scaler fit/transform, deliberately split into two functions.

fit_scaler is a training-only operation — it must only ever see train-
partition rows. transform_with_scaler applies an already-fitted scaler
and has no concept of splits at all, which is what makes it usable
unmodified at serving time (a live row is neither train, val, nor test).
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_scaler(
    df: pd.DataFrame,
    columns: list[str],
    train_mask: pd.Series,
) -> StandardScaler:
    """
    Fit a StandardScaler using only rows where train_mask is True.

    Never call this with val/test/live data — the whole point of a
    train-only fit is that val/test statistics must not influence the
    learned mean/std (see Phase 2 leakage demo: this is what prevents
    future distribution shift from leaking into training data).
    """
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, columns])
    return scaler


def transform_with_scaler(
    df: pd.DataFrame,
    scaler: StandardScaler,
    columns: list[str],
) -> pd.DataFrame:
    """
    Apply an already-fitted scaler to `columns` in `df`.

    Never fits — this function has no mask parameter and no concept of
    train/val/test/serving. It transforms whatever rows are in `df`
    using the scaler's frozen statistics. This is what makes it directly
    reusable, unmodified, in Phase 6 serving code.
    """
    df = df.copy()
    df[columns] = scaler.transform(df[columns])
    return df
