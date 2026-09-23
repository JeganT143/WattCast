import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    column: str,
    lags: list[int],
) -> pd.DataFrame:
    """
    Add lag features for a column without mutating the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Chronologically ordered input DataFrame.
    column : str
        Column to create lag features from.
    lags : list[int]
        Number of previous timesteps to use.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with lag columns added.

    Raises
    ------
    ValueError
        If the DataFrame index is not chronologically ordered.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be chronologically ordered.")

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    features_df = df.copy()

    for lag in lags:
        features_df[f"lag_{lag}"] = features_df[column].shift(lag)

    return features_df
