import numpy as np
import pytest
import torch

from src.models.sequence_forecaster import SequenceForecaster, _COLUMNS


class _TinyNet(torch.nn.Module):
    def __init__(self, in_features, window):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(window * in_features, 8),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(8, 1)

    def forward(self, x):
        return self.head(self.body(x)).squeeze(-1)


class _Tiny(SequenceForecaster):
    calls = []
    should_boom = False

    def _build_network(self, n_features):
        if type(self).should_boom:
            raise RuntimeError("boom")
        type(self).calls.append((n_features, torch.initial_seed()))
        return _TinyNet(n_features, self.L)


def _make_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 16))
    y = rng.normal(100.0, 10.0, size=120)
    return X, y


def _tiny(**overrides):
    kwargs = dict(max_epochs=1, huber_delta=40.0, seed=7, L=6, batch_size=32)
    kwargs.update(overrides)
    return _Tiny(**kwargs)


def test_a_abstract_base_cannot_be_instantiated():
    with pytest.raises(TypeError):
        SequenceForecaster(max_epochs=1, huber_delta=40.0, seed=1)

    class _NoBuild(SequenceForecaster):
        pass

    with pytest.raises(TypeError):
        _NoBuild(max_epochs=1, huber_delta=40.0, seed=1)


def test_b_end_to_end_fit_predict_shapes():
    X, y = _make_data()
    assert X.shape[1] == len(_COLUMNS)

    m = _tiny()
    m.fit(X, y)
    pred = m.predict(X)

    assert pred.shape == (120,)
    assert np.isnan(pred[:5]).all()
    assert np.isfinite(pred[5:]).all()
    assert m.required_history_length == 5
    assert len(m.required_columns) == 16
    assert m.required_columns == list(_COLUMNS)


def test_c_rng_order_and_single_build():
    X, y = _make_data()
    _Tiny.calls = []
    m = _tiny()
    m.fit(X, y)
    assert len(_Tiny.calls) == 1
    n_features, seed_seen = _Tiny.calls[0]
    assert n_features == 16
    assert seed_seen == 7

    m.fit(X, y)
    assert len(_Tiny.calls) == 2


def test_d_failed_refit_leaves_state_untouched():
    X, y = _make_data()
    _Tiny.calls = []
    _Tiny.should_boom = False
    m = _tiny()
    m.fit(X, y)
    p0 = m.predict(X)
    losses0 = list(m.training_losses_)

    _Tiny.should_boom = True
    try:
        with pytest.raises(RuntimeError, match="boom"):
            m.fit(X, y)
    finally:
        _Tiny.should_boom = False

    p1 = m.predict(X)
    assert np.array_equal(p0, p1, equal_nan=True)
    assert m.training_losses_ == losses0


def test_e_head_contract():
    X, y = _make_data()

    class _NoHead(SequenceForecaster):
        def _build_network(self, n_features):
            class _M(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.body = torch.nn.Linear(1, 1)

                def forward(self, x):
                    return x

            return _M()

    m1 = _NoHead(max_epochs=1, huber_delta=40.0, seed=7, L=6, batch_size=32)
    with pytest.raises(TypeError):
        m1.fit(X, y)
    with pytest.raises(RuntimeError, match="call fit\\(\\) before predict\\(\\)"):
        m1.predict(X)

    class _BadHead(SequenceForecaster):
        def _build_network(self, n_features):
            class _M(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.head = torch.nn.Identity()

                def forward(self, x):
                    return x

            return _M()

    m2 = _BadHead(max_epochs=1, huber_delta=40.0, seed=7, L=6, batch_size=32)
    with pytest.raises(TypeError):
        m2.fit(X, y)
    with pytest.raises(RuntimeError, match="call fit\\(\\) before predict\\(\\)"):
        m2.predict(X)


def test_f_threads_restored():
    X, y = _make_data()
    original = torch.get_num_threads()
    torch.set_num_threads(3)
    try:
        m = _tiny()
        m.fit(X, y)
        assert torch.get_num_threads() == 3
        m.predict(X)
        assert torch.get_num_threads() == 3
    finally:
        torch.set_num_threads(original)


def test_g_params_and_validation():
    m = _tiny()
    assert set(m.params.keys()) == {
        "L", "hidden_size", "num_layers", "dropout", "learning_rate",
        "batch_size", "max_epochs", "loss", "huber_delta", "output_bias_init",
        "seed", "torch_num_threads", "optimizer",
    }

    with pytest.raises(ValueError):
        _tiny(huber_delta=0.0)
    with pytest.raises(ValueError):
        _tiny(dropout=0.5, num_layers=1)
    with pytest.raises(ValueError):
        _tiny(L=0)
    with pytest.raises(ValueError):
        _tiny(loss="mse")


def test_h_causality_non_vacuous():
    X, y = _make_data()
    m = _tiny()
    m.fit(X, y)
    pred0 = m.predict(X)
    X2 = X.copy()
    X2[40:] += 100.0
    pred1 = m.predict(X2)
    assert np.array_equal(pred0[:40], pred1[:40], equal_nan=True)
    assert not np.array_equal(pred0[40:], pred1[40:])


def test_i_determinism():
    X, y = _make_data()
    m1 = _tiny(seed=7)
    m1.fit(X, y)
    p1 = m1.predict(X)

    m2 = _tiny(seed=7)
    m2.fit(X, y)
    p2 = m2.predict(X)

    assert np.array_equal(p1, p2, equal_nan=True)
    assert m1.training_losses_ == m2.training_losses_

    m3 = _tiny(seed=8)
    m3.fit(X, y)
    p3 = m3.predict(X)
    assert not np.array_equal(p1, p3, equal_nan=True)
