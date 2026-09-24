"""Tests for src.evaluation.walk_forward.run_walk_forward — the walk-forward
runner, an orchestrator that wraps evaluate_on_validation per fold/model/seed
and adds no evaluation logic of its own."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from config.features import SPLIT_VAL_END, TARGET_HORIZONS
from config.paths import RAW_DATA_PATH
from config.walk_forward import EVAL_DAYS, FIRST_EVAL_START, FREQ_MINUTES, N_FOLDS
from src.evaluation.fold_data import build_fold_datasets
from src.evaluation.folds import make_folds
from src.evaluation.validation_only import evaluate_on_validation
from src.evaluation.walk_forward import run_walk_forward
from src.features.build_features import build_features
from src.models.forecaster import Forecaster
from src.models.naive import NaivePersistenceForecaster
from src.models.sklearn_models import LinearRegressionForecaster
from src.pipeline import run_pipeline


class _Fake(Forecaster):
    def __init__(self, seed=None, history=0, log=None):
        self.seed = seed
        self.history = history
        self.log = log

    @property
    def required_columns(self):
        return ["Appliances_lag_1"]

    @property
    def required_history_length(self):
        return self.history

    def fit(self, X, y):
        if self.log is not None:
            self.log.append(id(self))
        return self

    def predict(self, X):
        preds = X[:, 0].astype(np.float64).copy()
        preds[: self.history] = np.nan
        return preds

    @property
    def params(self):
        return {"seed": self.seed, "history": self.history}


@pytest.fixture(scope="module")
def df_raw():
    return pd.read_csv(RAW_DATA_PATH, parse_dates=["date"])


@pytest.fixture(scope="module")
def df_features(df_raw):
    return build_features(df_raw, target_horizons=TARGET_HORIZONS)


@pytest.fixture(scope="module")
def result(df_raw):
    return run_pipeline(df_raw)


@pytest.fixture(scope="module")
def specs(result):
    timestamps = pd.DatetimeIndex(
        pd.concat(
            [
                result.train_t6["date"],
                result.val_t6["date"],
                result.test_t6["date"],
            ],
            ignore_index=True,
        )
    )
    return make_folds(
        timestamps,
        first_eval_start=FIRST_EVAL_START,
        n_folds=N_FOLDS,
        eval_days=EVAL_DAYS,
        freq_minutes=FREQ_MINUTES,
        holdout_start=pd.Timestamp(SPLIT_VAL_END),
    )


@pytest.fixture(scope="module")
def base_records(df_features, specs):
    return run_walk_forward(
        df_features,
        specs,
        horizon=6,
        deterministic={
            "naive_persistence": NaivePersistenceForecaster,
            "linear_regression": LinearRegressionForecaster,
        },
        seeded={"fake_seeded": lambda seed: _Fake(seed=seed, history=3)},
    )


def test_a_record_shape_and_ordering(base_records):
    assert len(base_records) == 8 * (2 + 3)

    triples = [(r.fold, r.model, r.seed) for r in base_records]
    assert len(set(triples)) == len(triples)

    folds = [r.fold for r in base_records]
    assert folds == sorted(folds)

    det_records = [r for r in base_records if r.model in ("naive_persistence", "linear_regression")]
    assert all(r.seed is None for r in det_records)

    seeded_records = [r for r in base_records if r.model == "fake_seeded"]
    for fold in {r.fold for r in seeded_records}:
        seeds_for_fold = sorted(r.seed for r in seeded_records if r.fold == fold)
        assert seeds_for_fold == [42, 43, 44]


def test_b_factory_freshness(df_features, specs):
    det_calls = []
    seeded_calls = []
    fit_log = []
    instances = []

    def det_factory():
        det_calls.append(1)
        forecaster = _Fake(log=fit_log)
        instances.append(forecaster)
        return forecaster

    def seeded_factory(seed):
        seeded_calls.append(seed)
        forecaster = _Fake(seed=seed, log=fit_log)
        instances.append(forecaster)
        return forecaster

    run_walk_forward(
        df_features,
        specs,
        horizon=6,
        deterministic={"fake_det": det_factory},
        seeded={"fake_seeded": seeded_factory},
    )

    assert len(det_calls) == 8
    assert seeded_calls == [42, 43, 44] * 8
    assert len(fit_log) == 32
    assert len(set(fit_log)) == 32


def test_c_independent_persistence_arithmetic(df_raw, specs, base_records):
    spec3 = specs[2]
    y = df_raw["Appliances"]
    mask = (df_raw["date"] >= spec3.eval_start) & (df_raw["date"] < spec3.eval_end)
    diff = (y.shift(-6) - y)[mask]

    assert not diff.isna().any()

    record = next(
        r for r in base_records if r.fold == spec3.fold and r.model == "naive_persistence"
    )
    expected_mae = np.mean(np.abs(diff))
    expected_rmse = np.sqrt(np.mean(diff**2))

    assert record.mae == pytest.approx(expected_mae, rel=1e-12)
    assert record.rmse == pytest.approx(expected_rmse, rel=1e-12)


def test_d_linear_regression_oracle(df_features, specs, base_records):
    spec3 = specs[2]
    fd = build_fold_datasets(df_features, spec3, 6)
    oracle = evaluate_on_validation(
        LinearRegressionForecaster(), fd.train_df, fd.eval_df, "target_t6"
    )

    record = next(
        r for r in base_records if r.fold == spec3.fold and r.model == "linear_regression"
    )
    assert record.mae == pytest.approx(oracle.metrics["mae"], rel=1e-12)
    assert record.rmse == pytest.approx(oracle.metrics["rmse"], rel=1e-12)


def test_e_causality_tripwire(df_features, specs):
    spec3 = specs[2]
    mutated = df_features.copy()
    numeric_cols = [c for c in mutated.columns if c != "date" and pd.api.types.is_numeric_dtype(mutated[c])]
    future_mask = mutated["date"] >= spec3.eval_end
    mutated.loc[future_mask, numeric_cols] = mutated.loc[future_mask, numeric_cols] * 1000

    det = {
        "naive_persistence": NaivePersistenceForecaster,
        "linear_regression": LinearRegressionForecaster,
    }

    base = run_walk_forward(
        df_features, specs, horizon=6, deterministic=det, seeded={}, seeds=(42,)
    )
    mutated_records = run_walk_forward(
        mutated, specs, horizon=6, deterministic=det, seeded={}, seeds=(42,)
    )

    def by_key(records):
        return {(r.fold, r.model): r for r in records}

    base_by = by_key(base)
    mutated_by = by_key(mutated_records)

    for fold in (1, 2, 3):
        for model in ("naive_persistence", "linear_regression"):
            b = base_by[(fold, model)]
            m = mutated_by[(fold, model)]
            assert b.mae == pytest.approx(m.mae, rel=1e-12)
            assert b.rmse == pytest.approx(m.rmse, rel=1e-12)

    fold4_base = base_by[(4, "naive_persistence")]
    fold4_mutated = mutated_by[(4, "naive_persistence")]
    assert fold4_base.mae != pytest.approx(fold4_mutated.mae, rel=1e-12)


def test_f_callback_order(df_features, specs):
    seen = []
    records = run_walk_forward(
        df_features,
        specs,
        horizon=6,
        deterministic={
            "naive_persistence": NaivePersistenceForecaster,
            "linear_regression": LinearRegressionForecaster,
        },
        seeded={"fake_seeded": lambda seed: _Fake(seed=seed, history=3)},
        on_record=lambda r: seen.append(r),
    )
    assert tuple(seen) == records


def test_g_validation(df_features, specs):
    det = {"naive_persistence": NaivePersistenceForecaster}
    seeded = {"fake_seeded": lambda seed: _Fake(seed=seed)}

    with pytest.raises(ValueError):
        run_walk_forward(df_features, (), horizon=6, deterministic=det, seeded=seeded)

    with pytest.raises(ValueError):
        run_walk_forward(
            df_features, specs, horizon=6, deterministic=det, seeded=seeded, seeds=()
        )

    with pytest.raises(ValueError):
        run_walk_forward(df_features, specs, horizon=6, deterministic={}, seeded={})

    with pytest.raises(ValueError):
        run_walk_forward(
            df_features,
            specs,
            horizon=6,
            deterministic={"dup": NaivePersistenceForecaster},
            seeded={"dup": lambda seed: _Fake(seed=seed)},
        )


def test_h_record_contract(df_features, specs, base_records):
    for r in base_records:
        assert r.n_evaluated == 1008
        assert r.n_ineligible_label_rows == 6
        if r.model == "fake_seeded":
            assert r.n_warmup_rows == 3
        else:
            assert r.n_warmup_rows == 0
        assert np.isfinite(r.mae)
        assert np.isfinite(r.rmse)
        assert np.isfinite(r.mape)
        assert r.mape > 0

    record = base_records[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.mae = 0.0

    before = df_features.copy(deep=True)
    run_walk_forward(
        df_features,
        specs,
        horizon=6,
        deterministic={
            "naive_persistence": NaivePersistenceForecaster,
            "linear_regression": LinearRegressionForecaster,
        },
        seeded={"fake_seeded": lambda seed: _Fake(seed=seed, history=3)},
    )
    pd.testing.assert_frame_equal(df_features, before)
