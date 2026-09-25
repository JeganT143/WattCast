import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.validation_only import evaluate_on_validation
from src.models.cnn_lstm import CNNLSTMForecaster, _CNNLSTMNet
from src.models.gru import GRUForecaster
from src.models.lstm import LSTMForecaster
from src.models.sequence_forecaster import SequenceForecaster


def _make_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 16))
    y = rng.normal(100.0, 10.0, size=120)
    return X, y


def _cnn_lstm(**overrides):
    kwargs = dict(max_epochs=1, huber_delta=40.0, seed=7, L=6, batch_size=32)
    kwargs.update(overrides)
    return CNNLSTMForecaster(**kwargs)


def test_a_structure_and_params():
    assert issubclass(CNNLSTMForecaster, SequenceForecaster)

    m = CNNLSTMForecaster(max_epochs=1, huber_delta=40.0, seed=1)
    assert m.required_history_length == 17
    assert len(m.required_columns) == 16

    lstm = LSTMForecaster(max_epochs=1, huber_delta=40.0, seed=1)
    base_keys = set(lstm.params.keys())
    assert base_keys <= set(m.params.keys())
    assert set(m.params.keys()) - base_keys == {
        "conv_channels", "conv_kernel_size", "conv_padding", "conv_activation",
    }
    assert len(m.params) == 17
    assert m.params["conv_channels"] == 32
    assert m.params["conv_kernel_size"] == 3
    assert m.params["conv_padding"] == 1
    assert m.params["conv_activation"] == "relu"


def test_b_architecture_independent_arithmetic():
    X, y = _make_data()
    m = _cnn_lstm()
    m.fit(X, y)

    net = m._net
    assert isinstance(net.conv, torch.nn.Conv1d)
    assert net.conv.in_channels == 16
    assert net.conv.out_channels == 32
    assert net.conv.kernel_size == (3,)
    assert net.conv.padding == (1,)
    assert isinstance(net.act, torch.nn.ReLU)
    assert isinstance(net.lstm, torch.nn.LSTM)
    assert net.lstm.input_size == 32
    assert net.lstm.hidden_size == 64
    assert isinstance(net.head, torch.nn.Linear)

    conv_params = 32 * 16 * 3 + 32
    lstm_params = 4 * (64 * 32 + 64 * 64 + 64 + 64)
    head_params = 64 + 1
    total = conv_params + lstm_params + head_params
    assert total == 26721
    assert sum(p.numel() for p in net.parameters()) == total


def test_c_end_to_end_and_bias():
    X, y = _make_data()
    m = _cnn_lstm()
    m.fit(X, y)
    pred = m.predict(X)
    assert pred.shape == (120,)
    assert np.isnan(pred[:5]).all()
    assert np.isfinite(pred[5:]).all()
    assert m.initial_output_bias_ == float(np.median(y))


def test_d_window_length_preserved():
    net = _CNNLSTMNet(16, 64, 1, 0.0)
    t = torch.randn(4, 18, 16, dtype=torch.float32)
    conv_out = net.conv(t.transpose(1, 2))
    assert conv_out.shape == (4, 32, 18)
    out = net(t)
    assert out.shape == (4,)


def test_e_differs_from_others():
    X, y = _make_data()
    m = _cnn_lstm()
    m.fit(X, y)
    pred_cnn_lstm = m.predict(X)

    lstm = LSTMForecaster(max_epochs=1, huber_delta=40.0, seed=7, L=6, batch_size=32)
    lstm.fit(X, y)
    pred_lstm = lstm.predict(X)

    gru = GRUForecaster(max_epochs=1, huber_delta=40.0, seed=7, L=6, batch_size=32)
    gru.fit(X, y)
    pred_gru = gru.predict(X)

    assert not np.array_equal(pred_cnn_lstm[5:], pred_lstm[5:])
    assert not np.array_equal(pred_cnn_lstm[5:], pred_gru[5:])


def test_f_determinism():
    X, y = _make_data()
    m1 = _cnn_lstm(seed=7)
    m1.fit(X, y)
    p1 = m1.predict(X)

    m2 = _cnn_lstm(seed=7)
    m2.fit(X, y)
    p2 = m2.predict(X)

    assert np.array_equal(p1, p2, equal_nan=True)
    assert m1.training_losses_ == m2.training_losses_

    m3 = _cnn_lstm(seed=8)
    m3.fit(X, y)
    p3 = m3.predict(X)
    assert not np.array_equal(p1, p3, equal_nan=True)


def test_g_causality_non_vacuous():
    X, y = _make_data()
    m = _cnn_lstm()
    m.fit(X, y)
    pred0 = m.predict(X)
    X2 = X.copy()
    X2[40:] += 100.0
    pred1 = m.predict(X2)
    assert np.array_equal(pred0[:40], pred1[:40], equal_nan=True)
    assert not np.array_equal(pred0[40:], pred1[40:])


@pytest.mark.requires_processed_data
def test_h_real_data_smoke_validation_only():
    train_df = pd.read_csv("data/processed/train_t6.csv", parse_dates=["date"])
    val_df = pd.read_csv("data/processed/val_t6.csv", parse_dates=["date"])

    result = evaluate_on_validation(
        CNNLSTMForecaster(max_epochs=1, huber_delta=40.0, seed=42), train_df, val_df, "target_t6"
    )

    assert result.n_evaluated == 1728
    assert result.n_warmup_rows == 17
    assert result.n_ineligible_label_rows == 6
    assert np.isfinite(result.metrics["mae"]) and result.metrics["mae"] > 0
    assert np.isfinite(result.metrics["rmse"]) and result.metrics["rmse"] > 0
    assert result.metrics["rmse"] >= result.metrics["mae"]
