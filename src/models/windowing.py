"""One pure function, make_windows, that turns a (n, n_features) array into overlapping causal
windows. It is used by BOTH training (SequenceDataset) and inference (LSTMForecaster.predict) so
the two can never window differently. It knows nothing about targets, timestamps, partitions, or
models.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def make_windows(X: np.ndarray, L: int) -> np.ndarray:
    """Window i is X[i : i + L]; there are max(0, n - L + 1) of them; the last row of window i
    is row i + L - 1; the result is an independent copy (never a view of X); dtype is preserved;
    it never reads rows after a window's last row.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_rows, n_features), got shape {X.shape}")
    if L <= 0:
        raise ValueError(f"L must be positive, got {L}")

    n, f = X.shape
    if L > n:
        return np.empty((0, L, f), dtype=X.dtype)

    view = sliding_window_view(X, L, axis=0)
    return np.array(view.transpose(0, 2, 1), order="C", copy=True)
