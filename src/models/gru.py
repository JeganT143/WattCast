"""A PyTorch GRU regressor behind the Forecaster contract. It shares windowing, seeding,
threading and the training loop with LSTMForecaster through SequenceForecaster, and
differs only in the recurrent layer.
"""

import torch

from src.models.sequence_forecaster import SequenceForecaster


class _GRUNet(torch.nn.Module):
    def __init__(self, n_features, hidden_size, num_layers, dropout):
        super().__init__()
        self.gru = torch.nn.GRU(
            n_features, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout
        )
        self.head = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class GRUForecaster(SequenceForecaster):
    def _build_network(self, n_features):
        return _GRUNet(n_features, self.hidden_size, self.num_layers, self.dropout)
