"""A torch Dataset over causal windows built with make_windows, the same function
LSTMForecaster.predict uses, so training and inference can never window differently.
It knows nothing about partitions, timestamps, horizons, or purge rules: it receives
exactly the rows it is meant to expose. It owns its tensors (float32) and never shares
memory with the caller's arrays.
"""

import numpy as np
import torch

from src.models.windowing import make_windows


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, L):
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n_rows, n_features), got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")
        if L <= 0:
            raise ValueError(f"L must be positive, got {L}")
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length, got {len(X)} and {len(y)}")
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values")
        if not np.isfinite(y).all():
            raise ValueError("y contains non-finite values (only eligible rows may reach the dataset)")

        windows = make_windows(X, L)
        self._features = torch.from_numpy(windows).to(torch.float32)
        self._targets = torch.from_numpy(np.array(y[L - 1 :], dtype=np.float32))

    def __len__(self):
        return int(self._features.shape[0])

    def __getitem__(self, i):
        return self._features[i], self._targets[i]
