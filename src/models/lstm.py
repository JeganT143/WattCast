"""A PyTorch LSTM regressor behind the Forecaster contract. It consumes the same flat
(n, features) arrays as the sklearn wrappers; windowing is done here with make_windows
(the same function SequenceDataset uses). required_history_length is L - 1: the harness
prepends that many preceding rows and the model returns NaN for exactly the first L - 1
positions of its input.
"""

import torch

from src.models.sequence_forecaster import SequenceForecaster


class _Net(torch.nn.Module):
    def __init__(self, n_features, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            n_features, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout
        )
        self.head = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class LSTMForecaster(SequenceForecaster):
    def _build_network(self, n_features):
        return _Net(n_features, self.hidden_size, self.num_layers, self.dropout)
