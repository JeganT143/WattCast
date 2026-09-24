"""Base class behind PyTorch sequence regressors (LSTM and future variants) implementing
the Forecaster contract. It consumes the same flat (n, features) arrays as the sklearn
wrappers; windowing is done here with make_windows (the same function SequenceDataset
uses). required_history_length is L - 1: the harness prepends that many preceding rows
and the model returns NaN for exactly the first L - 1 positions of its input.

Concrete subclasses supply _build_network; everything else (constructor, validation,
windowing, seeding, threading, the training loop and predict) lives here.
"""

import math
import operator
from abc import abstractmethod
from contextlib import contextmanager

import numpy as np
import torch

from config.features import FEATURE_COLUMNS
from src.models.forecaster import Forecaster
from src.models.sequence_dataset import SequenceDataset
from src.models.windowing import make_windows

_EXCLUDED = {"hour_of_day", "day_of_week"}
_COLUMNS = [c for c in FEATURE_COLUMNS if c not in _EXCLUDED]


@contextmanager
def _threads(n):
    original = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(original)


class SequenceForecaster(Forecaster):
    def __init__(
        self,
        *,
        max_epochs,
        huber_delta,
        seed,
        L=18,
        hidden_size=64,
        num_layers=1,
        dropout=0.0,
        learning_rate=1e-3,
        batch_size=64,
        loss="huber",
        output_bias_init="train_target_median",
        torch_num_threads=4,
        optimizer="adam",
    ):
        L = operator.index(L)
        hidden_size = operator.index(hidden_size)
        num_layers = operator.index(num_layers)
        batch_size = operator.index(batch_size)
        max_epochs = operator.index(max_epochs)
        seed = operator.index(seed)
        torch_num_threads = operator.index(torch_num_threads)
        dropout = float(dropout)
        learning_rate = float(learning_rate)
        huber_delta = float(huber_delta)

        if loss != "huber":
            raise ValueError(f"loss must be 'huber' (the only registered value), got {loss!r}")
        if optimizer != "adam":
            raise ValueError(f"optimizer must be 'adam' (the only registered value), got {optimizer!r}")
        if output_bias_init != "train_target_median":
            raise ValueError(
                f"output_bias_init must be 'train_target_median' (the only registered value), got {output_bias_init!r}"
            )
        if not (huber_delta > 0) or not math.isfinite(huber_delta):
            raise ValueError(f"huber_delta must be a positive finite number, got {huber_delta}")
        if L < 1:
            raise ValueError(f"L must be positive, got {L}")
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}")
        if dropout > 0 and num_layers == 1:
            raise ValueError("dropout requires num_layers > 1")
        if not (learning_rate >= 0):
            raise ValueError(f"learning_rate must be non-negative, got {learning_rate}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if max_epochs < 1:
            raise ValueError(f"max_epochs must be positive, got {max_epochs}")
        if torch_num_threads < 1:
            raise ValueError(f"torch_num_threads must be positive, got {torch_num_threads}")

        self.L = L
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.loss = loss
        self.huber_delta = huber_delta
        self.output_bias_init = output_bias_init
        self.seed = seed
        self.torch_num_threads = torch_num_threads
        self.optimizer = optimizer

    @abstractmethod
    def _build_network(self, n_features: int) -> torch.nn.Module:
        """Build the network for this model: a float32 (B, L, n_features) batch in, (B,) out.

        The returned module must expose a torch.nn.Linear attribute named head — its bias
        is initialised to the training-target median before training starts.
        """
        ...

    @property
    def required_columns(self):
        return list(_COLUMNS)

    @property
    def required_history_length(self):
        return self.L - 1

    @property
    def params(self):
        return {
            "L": self.L,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "loss": self.loss,
            "huber_delta": self.huber_delta,
            "output_bias_init": self.output_bias_init,
            "seed": self.seed,
            "torch_num_threads": self.torch_num_threads,
            "optimizer": self.optimizer,
        }

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n_rows, n_features), got shape {X.shape}")
        if X.shape[1] != len(_COLUMNS):
            raise ValueError(f"expected {len(_COLUMNS)} columns, got {X.shape[1]}")
        if y.ndim != 1 or len(X) != len(y):
            raise ValueError(f"X and y must have the same length, got {len(X)} and {len(y)}")
        if len(X) < self.L:
            raise ValueError(f"fit needs at least {self.L} rows, got {len(X)}")
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values")
        if not np.isfinite(y).all():
            raise ValueError("y contains non-finite values")

        with _threads(self.torch_num_threads), torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            net = self._build_network(len(_COLUMNS))
            if not isinstance(getattr(net, "head", None), torch.nn.Linear):
                raise TypeError(
                    "_build_network must return a module with a torch.nn.Linear attribute "
                    "named head (its bias is initialised to the training-target median)"
                )
            bias = float(np.median(y))
            with torch.no_grad():
                net.head.bias.fill_(bias)
            dataset = SequenceDataset(X, y, self.L)
            g = torch.Generator()
            g.manual_seed(self.seed)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, generator=g
            )
            opt = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
            crit = torch.nn.HuberLoss(delta=self.huber_delta)
            losses = []
            net.train()
            for epoch in range(self.max_epochs):
                total = 0.0
                for xb, yb in loader:
                    opt.zero_grad()
                    pred = net(xb)
                    if pred.shape != yb.shape:
                        raise RuntimeError(
                            f"prediction shape {tuple(pred.shape)} != target shape {tuple(yb.shape)}"
                        )
                    loss = crit(pred, yb)
                    loss.backward()
                    opt.step()
                    total += float(loss.item()) * len(yb)
                epoch_loss = total / len(dataset)
                if not math.isfinite(epoch_loss):
                    raise RuntimeError(f"non-finite training loss at epoch {epoch}")
                losses.append(epoch_loss)
            net.eval()

        # assigned together, only after training has succeeded: a failed fit leaves a fitted model unchanged
        self._net = net
        self.training_losses_ = losses
        self.initial_output_bias_ = bias
        return self

    def predict(self, X):
        if not hasattr(self, "_net"):
            raise RuntimeError("call fit() before predict()")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n_rows, n_features), got shape {X.shape}")
        if X.shape[1] != len(_COLUMNS):
            raise ValueError(f"expected {len(_COLUMNS)} columns, got {X.shape[1]}")
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values")

        n = len(X)
        out = np.full(n, np.nan, dtype=np.float64)
        windows = make_windows(X, self.L)
        if len(windows) > 0:
            with _threads(self.torch_num_threads), torch.no_grad():
                t = torch.from_numpy(windows).to(torch.float32)
                preds = torch.cat([self._net(t[i : i + 4096]) for i in range(0, len(t), 4096)])
            out[self.L - 1 :] = preds.numpy().astype(np.float64)
        return out
