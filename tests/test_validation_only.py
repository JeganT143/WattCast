import dataclasses
import inspect
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.features import FEATURE_COLUMNS
from src.evaluation.harness import evaluate_forecaster
from src.evaluation.metrics import mae, rmse
from src.evaluation.validation_only import ValidationResult, evaluate_on_validation
from src.models.forecaster import Forecaster
from src.models.naive import NaivePersistenceForecaster, NaiveSeasonalForecaster
from src.models.sklearn_models import LinearRegressionForecaster, RandomForestForecaster

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "evaluation" / "validation_only.py"
PROCESSED = ROOT / "data" / "processed"
FIXTURE = ROOT / "tests" / "fixtures" / "golden_baselines.json"
TARGET = "target_t1"


def _series(n):
    i = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.date_range("2016-01-01", periods=n, freq="10min"),
            "Appliances": i * 10.0,
            TARGET: i * 10.0 + 5.0 + (np.arange(n) % 7) * 3.0,
        }
    )


def _split(n_train=10, n_val=6, n_test=5):
    df = _series(n_train + n_val + n_test)
    a, b = n_train, n_train + n_val
    return (
        df.iloc[:a].reset_index(drop=True),
        df.iloc[a:b].reset_index(drop=True),
        df.iloc[b:].reset_index(drop=True),
    )


def _with_ineligible_tail(train, n):
    train = train.copy()
    train.loc[train.index[-n:], TARGET] = np.nan
    return train


class _KBack(Forecaster):
    """Prediction at row i is the value k rows back; the first `nan_rows` rows are NaN (default k)."""

    def __init__(self, k, nan_rows=None):
        self.k = k
        self.nan_rows = k if nan_rows is None else nan_rows

    @property
    def required_columns(self):
        return ["Appliances"]

    @property
    def required_history_length(self):
        return self.k

    def fit(self, X, y):
        return self

    def predict(self, X):
        out = np.zeros(len(X))
        out[self.k :] = X[: len(X) - self.k, 0]
        out[: self.nan_rows] = np.nan
        return out

    @property
    def params(self):
        return {}


class _Recorder(_KBack):
    def __init__(self, k=0):
        super().__init__(k)
        self.fit_calls = 0
        self.fit_X = None
        self.fit_y = None
        self.predict_lengths = []

    def fit(self, X, y):
        self.fit_calls += 1
        self.fit_X, self.fit_y = X, y
        return self

    def predict(self, X):
        self.predict_lengths.append(len(X))
        return super().predict(X)


class _WrongShape(_KBack):
    def predict(self, X):
        return np.zeros((len(X), 1))


def _harness(model, train, val, test):
    return evaluate_forecaster(
        model, train, val, test, target_column=TARGET, model_name="stub", horizon=1, mape_threshold=1.0
    )


# ---------------------------------------------------------------------
# Interface and result structure
# ---------------------------------------------------------------------


def test_result_has_exactly_the_agreed_fields_and_only_validation_content():
    assert dataclasses.is_dataclass(ValidationResult)
    assert {f.name for f in dataclasses.fields(ValidationResult)} == {
        "metrics", "predictions", "actuals", "n_evaluated", "n_warmup_rows", "n_ineligible_label_rows",
    }
    train, val, _ = _split()
    got = evaluate_on_validation(_KBack(0), train, val, TARGET)
    assert isinstance(got, ValidationResult)
    assert set(got.metrics) == {"mae", "rmse"}
    for arr in (got.predictions, got.actuals):
        assert isinstance(arr, np.ndarray) and arr.ndim == 1 and arr.shape == (len(val),)
    for count in (got.n_evaluated, got.n_warmup_rows, got.n_ineligible_label_rows):
        assert type(count) is int


def test_signature_has_exactly_the_four_agreed_parameters():
    assert list(inspect.signature(evaluate_on_validation).parameters) == [
        "forecaster", "train_df", "val_df", "target_column",
    ]


# ---------------------------------------------------------------------
# Equivalence with the harness
# ---------------------------------------------------------------------


@pytest.mark.parametrize("k", [0, 3])
@pytest.mark.parametrize("n_inel", [0, 2])
def test_matches_the_harness_exactly_on_stubs(k, n_inel):
    train, val, test = _split(10, 6, 5)
    if n_inel:
        train = _with_ineligible_tail(train, n_inel)
    ref = _harness(_KBack(k), train, val, test)
    got = evaluate_on_validation(_KBack(k), train, val, TARGET)
    assert np.array_equal(got.predictions, ref.predictions["val"])
    assert np.array_equal(got.actuals, ref.actuals["val"])
    assert got.metrics["mae"] == ref.metrics["val"]["mae"]
    assert got.metrics["rmse"] == ref.metrics["val"]["rmse"]
    assert got.n_evaluated == ref.n_evaluated["val"]
    assert got.n_warmup_rows == ref.n_warmup_rows
    assert got.n_ineligible_label_rows == ref.n_ineligible_label_rows


def test_first_validation_prediction_uses_the_last_train_rows_even_when_their_labels_are_ineligible():
    train, val, _ = _split(10, 6, 5)
    train = _with_ineligible_tail(train, 2)
    got = evaluate_on_validation(_KBack(3), train, val, TARGET)
    assert got.predictions.shape == (6,)
    assert got.predictions[0] == 70.0  # Appliances of train row 7: the context is the FULL train frame
    assert got.n_evaluated == 6


def test_actuals_are_the_validation_labels_in_order():
    train, val, _ = _split(10, 6, 5)
    got = evaluate_on_validation(_KBack(3), _with_ineligible_tail(train, 2), val, TARGET)
    assert np.array_equal(got.actuals, val[TARGET].to_numpy())


def test_metrics_come_from_the_shared_metric_functions():
    train, val, _ = _split()
    got = evaluate_on_validation(_KBack(3), train, val, TARGET)
    assert got.metrics == {"mae": mae(got.actuals, got.predictions), "rmse": rmse(got.actuals, got.predictions)}


def test_linear_regression_matches_the_harness_bit_for_bit_on_synthetic_data():
    n_train, n_val, n_test, n_inel = 400, 1728, 50, 6
    n = n_train + n_val + n_test
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    df.insert(0, "date", pd.date_range("2016-01-01", periods=n, freq="10min"))
    df[TARGET] = 60.0 + 10.0 * df[FEATURE_COLUMNS[0]] + rng.normal(scale=3.0, size=n)
    train = _with_ineligible_tail(df.iloc[:n_train].reset_index(drop=True), n_inel)
    val = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test = df.iloc[n_train + n_val :].reset_index(drop=True)
    ref = _harness(LinearRegressionForecaster(), train, val, test)
    got = evaluate_on_validation(LinearRegressionForecaster(), train, val, TARGET)
    assert np.array_equal(got.predictions, ref.predictions["val"])
    assert got.metrics["mae"] == ref.metrics["val"]["mae"]
    assert got.metrics["rmse"] == ref.metrics["val"]["rmse"]


_BASELINES = {
    "naive_persistence": lambda h: NaivePersistenceForecaster(),
    "naive_seasonal": lambda h: NaiveSeasonalForecaster(h),
    "linear_regression": lambda h: LinearRegressionForecaster(),
    "random_forest": lambda h: RandomForestForecaster(),
}


@pytest.mark.parametrize("h", [1, 6])
@pytest.mark.parametrize("name", list(_BASELINES))
def test_four_baselines_reproduce_the_fixture_validation_metrics_and_counts(name, h):
    train_path, val_path = PROCESSED / f"train_t{h}.csv", PROCESSED / f"val_t{h}.csv"
    if not (train_path.exists() and val_path.exists() and FIXTURE.exists()):
        pytest.skip("processed data or golden fixture not present (run `make data`)")
    train = pd.read_csv(train_path, parse_dates=["date"])
    val = pd.read_csv(val_path, parse_dates=["date"])
    expected = json.loads(FIXTURE.read_text())["results"][f"{name}|h{h}"]
    got = evaluate_on_validation(_BASELINES[name](h), train, val, f"target_t{h}")
    assert got.n_evaluated == expected["n_evaluated"]["val"]
    assert got.n_evaluated == expected["n_rows"]["val"]
    assert got.n_warmup_rows == 0
    assert got.n_ineligible_label_rows == h
    for metric in ("mae", "rmse"):
        want = expected["metrics"][f"val_{metric}"]
        if name == "linear_regression":
            assert got.metrics[metric] == pytest.approx(want, rel=1e-12, abs=0), metric
        else:
            assert got.metrics[metric] == want, metric


# ---------------------------------------------------------------------
# Failure parity with the harness, and call discipline
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "case, match", [("val_nan", "non-finite"), ("non_trailing", "trailing"), ("overlap", "overlap")]
)
def test_invalid_label_configurations_raise_before_fit(case, match):
    train, val, _ = _split(10, 6, 5)
    k = 0
    if case == "val_nan":
        val = val.copy()
        val.loc[val.index[2], TARGET] = np.nan
    elif case == "non_trailing":
        train = train.copy()
        train.loc[train.index[4], TARGET] = np.nan
    else:
        train, val, _ = _split(4, 6, 5)
        train = _with_ineligible_tail(train, 2)
        k = 3
    model = _Recorder(k)
    with pytest.raises(ValueError, match=match):
        evaluate_on_validation(model, train, val, TARGET)
    assert model.fit_calls == 0


@pytest.mark.parametrize(
    "model, match",
    [
        (_KBack(k=3, nan_rows=2), "violates"),
        (_KBack(k=0, nan_rows=3), "violates"),
        (_WrongShape(0), "predict must return shape"),
    ],
)
def test_model_contract_violations_raise(model, match):
    train, val, _ = _split()
    with pytest.raises(ValueError, match=match):
        evaluate_on_validation(model, train, val, TARGET)


def test_a_gap_between_train_and_validation_is_rejected_when_context_is_needed():
    df = _series(21)
    train, val = df.iloc[:10].reset_index(drop=True), df.iloc[12:18].reset_index(drop=True)
    with pytest.raises(ValueError, match="not contiguous"):
        evaluate_on_validation(_KBack(3), train, val, TARGET)


def test_fits_exactly_once_on_the_eligible_rows_only():
    train, val, _ = _split(10, 6, 5)
    train = _with_ineligible_tail(train, 2)
    model = _Recorder(3)
    evaluate_on_validation(model, train, val, TARGET)
    assert model.fit_calls == 1
    assert len(model.fit_X) == 8 and len(model.fit_y) == 8
    assert np.isfinite(model.fit_y).all()
    assert np.array_equal(model.fit_X, train[["Appliances"]].to_numpy()[:8])
    assert np.array_equal(model.fit_y, train[TARGET].to_numpy()[:8])


@pytest.mark.parametrize("k", [0, 3])
def test_predict_is_called_once_on_context_plus_validation_rows_and_never_on_train(k):
    train, val, _ = _split(10, 6, 5)
    model = _Recorder(k)
    evaluate_on_validation(model, train, val, TARGET)
    assert model.predict_lengths == [k + len(val)]


def test_inputs_are_not_mutated():
    train, val, _ = _split()
    train = _with_ineligible_tail(train, 2)
    train0, val0 = train.copy(), val.copy()
    evaluate_on_validation(_KBack(3), train, val, TARGET)
    pd.testing.assert_frame_equal(train, train0)
    pd.testing.assert_frame_equal(val, val0)


# ---------------------------------------------------------------------
# Structural isolation: no test data, no file I/O, no harness, no tracking
# ---------------------------------------------------------------------


def test_evaluation_performs_no_file_io(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("file I/O attempted")

    train, val, _ = _split()
    monkeypatch.setattr(pd, "read_csv", boom)
    monkeypatch.setattr(Path, "read_text", boom)
    monkeypatch.setattr(Path, "open", boom)
    assert evaluate_on_validation(_KBack(3), train, val, TARGET).n_evaluated == len(val)


def _source():
    return MODULE.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "name, pattern",
    [
        ("test identifier", r"\btest_"),
        ("test csv path", r"test\.csv"),
        ("double-quoted test key", r'"test"'),
        ("single-quoted test key", r"'test'"),
        ("harness import", r"import.*harness|from .*harness"),
        ("mlflow", r"mlflow"),
        ("glob", r"\bglob\b|\.glob\(|import glob"),
        ("directory listing", r"iterdir|listdir|os\.walk"),
        ("file reading", r"read_csv|read_text|open\("),
    ],
)
def test_grep_gate_forbidden_pattern_is_absent(name, pattern):
    hits = [(i, line) for i, line in enumerate(_source().splitlines(), 1) if re.search(pattern, line)]
    assert not hits, (name, hits)


def test_module_imports_the_shared_helpers_from_context_and_metrics():
    src = _source()
    assert re.search(r"from src\.evaluation\.context import", src)
    assert re.search(r"from src\.evaluation\.metrics import", src)
    for name in ("take_context", "audit_leading_nans", "drop_context_predictions", "mae", "rmse"):
        assert re.search(rf"\b{name}\b", src), name


def test_module_does_not_reimplement_the_shared_helpers():
    pattern = r"^def (mae|rmse|take_context|audit_leading_nans|drop_context_predictions)\b"
    assert not re.search(pattern, _source(), re.M)
