import numpy as np
import pandas as pd
import pytest

from src.evaluation.validation_only import evaluate_on_validation
from src.models.gru import GRUForecaster
from src.models.lstm import LSTMForecaster
from src.models.sequence_forecaster import SequenceForecaster


def _make_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 16))
    y = rng.normal(100.0, 10.0, size=120)
    return X, y


def _gru(**overrides):
    kwargs = dict(max_epochs=1, huber_delta=40.0, seed=7, L=6, batch_size=32)
    kwargs.update(overrides)
    return GRUForecaster(**kwargs)


def test_a_structure():
    assert issubclass(GRUForecaster, SequenceForecaster)
    mro_names = [c.__name__ for c in GRUForecaster.__mro__]
    assert mro_names[:3] == ["GRUForecaster", "SequenceForecaster", "Forecaster"]

    m = GRUForecaster(max_epochs=1, huber_delta=40.0, seed=1)
    assert m.required_history_length == 17
    assert len(m.required_columns) == 16

    lstm = LSTMForecaster(max_epochs=1, huber_delta=40.0, seed=1)
    assert sorted(m.params.keys()) == sorted(lstm.params.keys())


def test_b_real_gru_independent_arithmetic():
    import torch

    X, y = _make_data()
    gru = _gru()
    gru.fit(X, y)
    assert isinstance(gru._net.gru, torch.nn.GRU)
    assert isinstance(gru._net.head, torch.nn.Linear)

    hidden = gru.hidden_size
    n_features = 16
    gru_params = 3 * (hidden * n_features + hidden * hidden + hidden + hidden) + (hidden + 1)
    assert gru_params == 15809
    assert sum(p.numel() for p in gru._net.parameters()) == gru_params

    lstm = LSTMForecaster(max_epochs=1, huber_delta=40.0, seed=7, L=6, batch_size=32)
    lstm.fit(X, y)
    lstm_params = 4 * (hidden * n_features + hidden * hidden + hidden + hidden) + (hidden + 1)
    assert lstm_params == 21057
    assert sum(p.numel() for p in lstm._net.parameters()) == lstm_params


def test_c_end_to_end_shapes():
    X, y = _make_data()
    m = _gru()
    m.fit(X, y)
    pred = m.predict(X)
    assert pred.shape == (120,)
    assert np.isnan(pred[:5]).all()
    assert np.isfinite(pred[5:]).all()


def test_d_differs_from_lstm():
    X, y = _make_data()
    gru = _gru()
    gru.fit(X, y)
    pred_gru = gru.predict(X)

    lstm = LSTMForecaster(max_epochs=1, huber_delta=40.0, seed=7, L=6, batch_size=32)
    lstm.fit(X, y)
    pred_lstm = lstm.predict(X)

    assert not np.array_equal(pred_gru[5:], pred_lstm[5:])


def test_e_determinism():
    X, y = _make_data()
    m1 = _gru(seed=7)
    m1.fit(X, y)
    p1 = m1.predict(X)

    m2 = _gru(seed=7)
    m2.fit(X, y)
    p2 = m2.predict(X)

    assert np.array_equal(p1, p2, equal_nan=True)
    assert m1.training_losses_ == m2.training_losses_

    m3 = _gru(seed=8)
    m3.fit(X, y)
    p3 = m3.predict(X)
    assert not np.array_equal(p1, p3, equal_nan=True)


def test_f_causality_non_vacuous():
    X, y = _make_data()
    m = _gru()
    m.fit(X, y)
    pred0 = m.predict(X)
    X2 = X.copy()
    X2[40:] += 100.0
    pred1 = m.predict(X2)
    assert np.array_equal(pred0[:40], pred1[:40], equal_nan=True)
    assert not np.array_equal(pred0[40:], pred1[40:])


def test_g_bias_contract():
    X, y = _make_data()
    m = _gru()
    m.fit(X, y)
    assert m.initial_output_bias_ == float(np.median(y))


@pytest.mark.requires_processed_data
def test_h_real_data_smoke_validation_only():
    train_df = pd.read_csv("data/processed/train_t6.csv", parse_dates=["date"])
    val_df = pd.read_csv("data/processed/val_t6.csv", parse_dates=["date"])

    result = evaluate_on_validation(
        GRUForecaster(max_epochs=1, huber_delta=40.0, seed=42), train_df, val_df, "target_t6"
    )

    assert result.n_evaluated == 1728
    assert result.n_warmup_rows == 17
    assert result.n_ineligible_label_rows == 6
    assert np.isfinite(result.metrics["mae"]) and result.metrics["mae"] > 0
    assert np.isfinite(result.metrics["rmse"]) and result.metrics["rmse"] > 0
    assert result.metrics["rmse"] >= result.metrics["mae"]
