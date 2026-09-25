import json

import numpy as np
import pandas as pd
import pytest
import torch

from config.features import FEATURE_COLUMNS, SCALED_COLUMNS
from src.evaluation.harness import evaluate_forecaster
from src.models.forecaster import Forecaster
from src.models.lstm import LSTMForecaster

EXPECTED_COLUMNS = [
    "Appliances_lag_1",
    "Appliances_lag_2",
    "Appliances_lag_3",
    "Appliances_lag_4",
    "Appliances_lag_5",
    "Appliances_lag_6",
    "Appliances_lag_144",
    "Appliances_roll6_mean",
    "Appliances_roll6_std",
    "Appliances_roll18_mean",
    "Appliances_roll18_std",
    "is_weekend",
    "minute_of_day_sin",
    "minute_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
]

PARAM_KEYS = {
    "L", "hidden_size", "num_layers", "dropout", "learning_rate", "batch_size", "max_epochs",
    "loss", "huber_delta", "output_bias_init", "seed", "torch_num_threads", "optimizer",
}


def _model(**overrides):
    kwargs = dict(
        L=4, hidden_size=8, num_layers=1, dropout=0.0, learning_rate=1e-3, batch_size=16,
        max_epochs=2, loss="huber", huber_delta=20.0, output_bias_init="train_target_median",
        seed=0, torch_num_threads=1, optimizer="adam",
    )
    kwargs.update(overrides)
    return LSTMForecaster(**kwargs)


def _xy(n=60, f=16, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f))
    y = 60.0 + 10.0 * X[:, 0] + rng.normal(scale=1.0, size=n)
    return X, y


# ---------------------------------------------------------------------
# Contract: identity, columns, history length, params
# ---------------------------------------------------------------------


def test_is_a_forecaster():
    assert isinstance(_model(), Forecaster)


def test_required_columns_are_the_sixteen_expected_columns_in_order():
    cols = _model().required_columns
    assert cols == EXPECTED_COLUMNS
    assert len(cols) == 16
    assert cols == [c for c in FEATURE_COLUMNS if c not in {"hour_of_day", "day_of_week"}]
    assert "is_weekend" in cols


def test_unscaled_required_columns_are_bounded_on_real_train_data():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data" / "processed" / "train_t6.csv"
    if not path.exists():
        pytest.skip("processed data not present (run `make data`)")
    df = pd.read_csv(path)
    unscaled = [c for c in _model().required_columns if c not in SCALED_COLUMNS]
    assert unscaled, "expected some unscaled columns"
    assert float(np.abs(df[unscaled].to_numpy()).max()) <= 1.0


@pytest.mark.parametrize("L", [1, 6, 18])
def test_required_history_length_is_L_minus_one(L):
    assert _model(L=L).required_history_length == L - 1


def test_defaults_match_the_registered_protocol():
    p = LSTMForecaster(max_epochs=10, huber_delta=20.0, seed=42).params
    assert p == {
        "L": 18, "hidden_size": 64, "num_layers": 1, "dropout": 0.0, "learning_rate": 1e-3,
        "batch_size": 64, "max_epochs": 10, "loss": "huber", "huber_delta": 20.0,
        "output_bias_init": "train_target_median", "seed": 42, "torch_num_threads": 4,
        "optimizer": "adam",
    }


def test_params_has_exactly_the_registered_keys_and_is_json_serializable():
    p = _model().params
    assert set(p) == PARAM_KEYS
    assert p["L"] == 4 and p["seed"] == 0 and p["huber_delta"] == 20.0
    json.dumps(p)


def test_params_returns_a_copy():
    m = _model()
    m.params["L"] = 999
    assert m.params["L"] == 4


def test_max_epochs_huber_delta_and_seed_have_no_defaults():
    with pytest.raises(TypeError):
        LSTMForecaster()


@pytest.mark.parametrize(
    "name, value",
    [
        ("loss", "mse"), ("optimizer", "sgd"), ("output_bias_init", "zero"),
        ("huber_delta", 0.0), ("huber_delta", -1.0), ("L", 0), ("hidden_size", 0),
        ("num_layers", 0), ("dropout", 1.0), ("dropout", -0.1), ("learning_rate", -1.0),
        ("batch_size", 0), ("max_epochs", 0), ("torch_num_threads", 0),
    ],
)
def test_invalid_constructor_values_raise(name, value):
    with pytest.raises(ValueError):
        _model(**{name: value})


# ---------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------


def test_fit_returns_self():
    X, y = _xy()
    m = _model()
    assert m.fit(X, y) is m


def test_fit_records_exactly_one_loss_per_epoch_and_no_early_stopping():
    X, y = _xy()
    m = _model(max_epochs=3).fit(X, y)
    assert len(m.training_losses_) == 3
    assert all(np.isfinite(v) for v in m.training_losses_)


def test_training_reduces_loss_on_a_learnable_problem():
    X, y = _xy(200)
    m = _model(hidden_size=16, learning_rate=1e-2, max_epochs=30, batch_size=32).fit(X, y)
    assert m.training_losses_[-1] < m.training_losses_[0]


def test_fit_raises_when_there_are_fewer_rows_than_L():
    X, y = _xy(3)
    with pytest.raises(ValueError, match="at least"):
        _model(L=4).fit(X, y)


def test_fit_raises_on_wrong_column_count():
    X, y = _xy()
    with pytest.raises(ValueError, match="columns"):
        _model().fit(X[:, :15], y)


def test_fit_raises_on_non_finite_targets():
    X, y = _xy()
    y[5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _model().fit(X, y)


def test_fit_does_not_mutate_its_inputs():
    X, y = _xy()
    X0, y0 = X.copy(), y.copy()
    _model().fit(X, y)
    assert np.array_equal(X, X0) and np.array_equal(y, y0)


def test_initial_output_bias_is_the_median_of_the_y_passed_to_fit():
    X, _ = _xy(60)
    y = np.arange(60.0)
    y[:3] = 1000.0
    m = _model().fit(X, y)
    assert m.initial_output_bias_ == float(np.median(y))
    assert m.initial_output_bias_ != float(np.median(y[3:]))


def test_with_a_zero_learning_rate_predictions_stay_within_the_head_scale_of_the_median():
    X, _ = _xy(60)
    y = np.full(60, 60.0)
    m = _model(hidden_size=8, learning_rate=0.0, max_epochs=1).fit(X, y)
    pred = m.predict(X)[3:]
    # default Linear init: |w_i| <= 1/sqrt(8) and |h_i| < 1, so |w.h| <= sqrt(8) ~ 2.83; the bias is the median
    assert float(np.abs(pred - 60.0).max()) < 3.0


# ---------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------


def test_predict_raises_before_fit():
    X, _ = _xy()
    with pytest.raises(RuntimeError, match="fit"):
        _model().predict(X)


def test_predict_raises_on_wrong_column_count():
    X, y = _xy()
    m = _model().fit(X, y)
    with pytest.raises(ValueError, match="columns"):
        m.predict(X[:, :15])


def test_predict_returns_float64_with_exactly_the_first_L_minus_one_positions_nan():
    X, y = _xy()
    m = _model(L=4).fit(X, y)
    out = m.predict(_xy(30, seed=1)[0])
    assert out.shape == (30,)
    assert out.dtype == np.float64
    assert np.isnan(out[:3]).all()
    assert np.isfinite(out[3:]).all()


@pytest.mark.parametrize("n", [0, 1, 3])
def test_predict_on_fewer_rows_than_L_is_all_nan(n):
    X, y = _xy()
    m = _model(L=4).fit(X, y)
    out = m.predict(_xy(max(n, 1), seed=2)[0][:n])
    assert out.shape == (n,)
    assert np.isnan(out).all()


def test_predictions_agree_when_the_input_is_shifted():
    X, y = _xy(80)
    m = _model(L=4, max_epochs=3).fit(X, y)
    Xn = _xy(40, seed=3)[0]
    full = m.predict(Xn)
    for s in (5, 12):
        part = m.predict(Xn[s:])
        np.testing.assert_allclose(part[3:], full[s + 3 :], rtol=1e-5, atol=1e-4)


def test_prediction_at_t_ignores_later_rows_and_depends_on_row_t():
    X, y = _xy(80)
    m = _model(L=4, max_epochs=3).fit(X, y)
    Xn = _xy(30, seed=4)[0]
    base = m.predict(Xn)

    future = Xn.copy()
    future[16:] += 50.0
    np.testing.assert_allclose(m.predict(future)[:16], base[:16], rtol=0, atol=1e-5)

    now = Xn.copy()
    now[15] += 5.0
    changed = m.predict(now)
    np.testing.assert_allclose(changed[:15], base[:15], rtol=0, atol=1e-5)
    assert abs(changed[15] - base[15]) > 1e-6


# ---------------------------------------------------------------------
# Reproducibility and process hygiene
# ---------------------------------------------------------------------


def test_same_seed_gives_identical_predictions():
    X, y = _xy()
    Xp = _xy(30, seed=1)[0]
    a = _model(seed=7).fit(X, y).predict(Xp)
    b = _model(seed=7).fit(X, y).predict(Xp)
    np.testing.assert_array_equal(a, b)


def test_refitting_the_same_instance_reproduces_the_same_predictions():
    X, y = _xy()
    Xp = _xy(30, seed=1)[0]
    m = _model(seed=7)
    a = m.fit(X, y).predict(Xp)
    b = m.fit(X, y).predict(Xp)
    np.testing.assert_array_equal(a, b)


def test_result_does_not_depend_on_global_torch_or_numpy_rng_state():
    X, y = _xy()
    Xp = _xy(30, seed=1)[0]
    torch.manual_seed(1)
    np.random.seed(1)
    a = _model(seed=5).fit(X, y).predict(Xp)
    torch.manual_seed(2)
    np.random.seed(2)
    b = _model(seed=5).fit(X, y).predict(Xp)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_give_different_predictions():
    X, y = _xy()
    Xp = _xy(30, seed=1)[0]
    a = _model(seed=0).fit(X, y).predict(Xp)
    b = _model(seed=1).fit(X, y).predict(Xp)
    assert not np.allclose(a[3:], b[3:])


def test_thread_count_is_applied_during_fit_and_restored_afterwards(monkeypatch):
    original = torch.get_num_threads()
    wanted = 3 if original != 3 else 2
    calls = []
    real = torch.set_num_threads

    def spy(n):
        calls.append(n)
        return real(n)

    monkeypatch.setattr(torch, "set_num_threads", spy)
    X, y = _xy()
    m = _model(torch_num_threads=wanted).fit(X, y)
    m.predict(X)
    assert wanted in calls
    assert torch.get_num_threads() == original


# ---------------------------------------------------------------------
# End to end through the evaluation harness
# ---------------------------------------------------------------------


def test_end_to_end_through_the_harness_with_warmup_and_ineligible_rows():
    n_train, n_val, n_test, n_inel = 40, 12, 10, 2
    n = n_train + n_val + n_test
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(n, 16)), columns=EXPECTED_COLUMNS)
    df.insert(0, "date", pd.date_range("2016-01-01", periods=n, freq="10min"))
    df["target_t6"] = 60.0 + 10.0 * df[EXPECTED_COLUMNS[0]].to_numpy() + rng.normal(scale=1.0, size=n)
    train = df.iloc[:n_train].reset_index(drop=True)
    val = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test = df.iloc[n_train + n_val :].reset_index(drop=True)
    train.loc[train.index[-n_inel:], "target_t6"] = np.nan

    result = evaluate_forecaster(
        _model(L=4), train, val, test,
        target_column="target_t6", model_name="lstm", horizon=6, mape_threshold=30.0,
    )
    assert result.n_warmup_rows == 3
    assert result.n_ineligible_label_rows == n_inel
    assert result.n_evaluated == {"train": n_train - 3 - n_inel, "val": n_val, "test": n_test}
    for partition in ("train", "val", "test"):
        assert np.isfinite(result.predictions[partition]).all()
        assert all(np.isfinite(v) for v in result.metrics[partition].values())


# ---------------------------------------------------------------------
# Rejected inputs, and a failed refit must not leave a half-updated model
# ---------------------------------------------------------------------


def test_dropout_with_a_single_layer_is_rejected():
    with pytest.raises(ValueError, match="num_layers"):
        _model(dropout=0.5, num_layers=1)


def test_dropout_with_two_layers_is_accepted_and_recorded():
    m = _model(dropout=0.5, num_layers=2)
    assert m.params["dropout"] == 0.5
    assert m.params["num_layers"] == 2


def test_predict_raises_on_non_finite_features():
    X, y = _xy()
    m = _model().fit(X, y)
    bad = _xy(30, seed=1)[0]
    bad[10, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        m.predict(bad)


def _overflowing_targets(n=60):
    # half +3e38, half -3e38: finite in float32, median 0, and every Huber term overflows float32 to inf
    return np.where(np.arange(n) % 2 == 0, 3.0e38, -3.0e38)


def test_non_finite_training_loss_raises():
    X, _ = _xy(60)
    with pytest.raises(RuntimeError, match="non-finite training loss"):
        _model().fit(X, _overflowing_targets())


@pytest.mark.parametrize(
    "name", ["L", "hidden_size", "num_layers", "batch_size", "max_epochs", "seed", "torch_num_threads"]
)
def test_integer_parameters_reject_floats(name):
    with pytest.raises(TypeError):
        _model(**{name: 2.5})


def test_required_columns_returns_a_fresh_list():
    m = _model()
    m.required_columns.append("junk")
    assert m.required_columns == EXPECTED_COLUMNS


def test_a_failed_refit_leaves_the_fitted_model_unchanged():
    X, y = _xy()
    m = _model(seed=3).fit(X, y)
    Xp = _xy(30, seed=1)[0]
    before = m.predict(Xp)
    bias_before = m.initial_output_bias_
    losses_before = list(m.training_losses_)
    with pytest.raises(RuntimeError):
        m.fit(X, _overflowing_targets())
    np.testing.assert_array_equal(m.predict(Xp), before)
    assert m.initial_output_bias_ == bias_before
    assert m.training_losses_ == losses_before
