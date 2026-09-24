"""A PyTorch CNN-LSTM regressor behind the Forecaster contract. The registered architecture
is one Conv1d of 32 channels, kernel 3, padding 1 (so the window length is preserved),
ReLU, then an LSTM with hidden size 64 and a Linear head on the last timestep. Every
position of a window is at or before the forecast origin, so the padding cannot leak
future rows. It shares windowing, seeding, threading and the training loop with the
other sequence models through SequenceForecaster.
"""

import torch

from src.models.sequence_forecaster import SequenceForecaster

CONV_CHANNELS = 32
CONV_KERNEL_SIZE = 3
CONV_PADDING = 1
CONV_ACTIVATION = "relu"


class _CNNLSTMNet(torch.nn.Module):
    def __init__(self, n_features, hidden_size, num_layers, dropout):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            n_features, CONV_CHANNELS, kernel_size=CONV_KERNEL_SIZE, padding=CONV_PADDING
        )
        self.act = torch.nn.ReLU()
        self.lstm = torch.nn.LSTM(
            CONV_CHANNELS, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout
        )
        self.head = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        z = self.act(self.conv(x.transpose(1, 2)))
        out, _ = self.lstm(z.transpose(1, 2))
        return self.head(out[:, -1, :]).squeeze(-1)


class CNNLSTMForecaster(SequenceForecaster):
    def _build_network(self, n_features):
        return _CNNLSTMNet(n_features, self.hidden_size, self.num_layers, self.dropout)

    @property
    def params(self):
        return {
            **super().params,
            "conv_channels": CONV_CHANNELS,
            "conv_kernel_size": CONV_KERNEL_SIZE,
            "conv_padding": CONV_PADDING,
            "conv_activation": CONV_ACTIVATION,
        }
