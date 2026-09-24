import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from config.features import SPLIT_VAL_END, TARGET_HORIZONS
from config.paths import RAW_DATA_PATH
from config.walk_forward import EVAL_DAYS, FIRST_EVAL_START, FREQ_MINUTES, N_FOLDS, NEURAL_SEEDS
from src.evaluation.folds import make_folds
from src.evaluation.walk_forward import FoldRunRecord
from src.evaluation.walk_forward_run import RECORDS_NAME, SUMMARY_NAME, execute, main, write_summary
from src.features.build_features import build_features
from src.models.forecaster import Forecaster
from src.models.naive import NaivePersistenceForecaster
from src.pipeline import run_pipeline


class _Fake(Forecaster):
    @property
    def required_columns(self):
        return ["Appliances_lag_1"]

    @property
    def required_history_length(self):
        return 0

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X[:, 0].astype(np.float64)

    @property
    def params(self):
        return {}


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
            [result.train_t6["date"], result.val_t6["date"], result.test_t6["date"]],
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


@pytest.fixture
def specs2(specs):
    return specs[:2]


def test_a_execute_returns_and_writes_records(df_features, specs2, tmp_path):
    records = execute(
        df_features,
        specs2,
        horizon=6,
        deterministic={"naive_persistence": NaivePersistenceForecaster},
        seeded={"fake": lambda seed: _Fake()},
        records_path=tmp_path / "recs.jsonl",
    )
    assert len(records) == 2 * (1 + 3)

    lines = (tmp_path / "recs.jsonl").read_text().splitlines()
    assert len(lines) == 8
    for line, record in zip(lines, records):
        assert json.loads(line) == dataclasses.asdict(record)


def test_b_incremental_write(df_features, specs2, tmp_path):
    records_path = tmp_path / "recs.jsonl"
    seen = []

    def on_progress(record):
        seen.append(record)
        lines = records_path.read_text().splitlines()
        assert len(lines) == len(seen)

    execute(
        df_features,
        specs2,
        horizon=6,
        deterministic={"naive_persistence": NaivePersistenceForecaster},
        seeded={"fake": lambda seed: _Fake()},
        records_path=records_path,
        on_progress=on_progress,
    )
    assert len(seen) == 8


def test_c_refuses_to_overwrite(df_features, specs2, tmp_path):
    records_path = tmp_path / "recs.jsonl"
    records_path.write_text("sentinel\n")

    det_calls = []
    seed_calls = []

    def det_factory():
        det_calls.append(1)
        return NaivePersistenceForecaster()

    def seed_factory(seed):
        seed_calls.append(seed)
        return _Fake()

    with pytest.raises(FileExistsError):
        execute(
            df_features,
            specs2,
            horizon=6,
            deterministic={"naive_persistence": det_factory},
            seeded={"fake": seed_factory},
            records_path=records_path,
        )

    assert records_path.read_text() == "sentinel\n"
    assert det_calls == []
    assert seed_calls == []


def _synthetic_records():
    records = []
    for fold in range(1, N_FOLDS + 1):
        for model in ("linear_regression", "random_forest", "naive_persistence", "naive_seasonal"):
            records.append(
                FoldRunRecord(
                    fold=fold, model=model, seed=None,
                    mae=100.0, rmse=200.0, mape=30.0,
                    n_evaluated=1008, n_warmup_rows=0, n_ineligible_label_rows=6,
                )
            )
        for model in ("lstm", "gru", "cnn_lstm"):
            for seed in NEURAL_SEEDS:
                records.append(
                    FoldRunRecord(
                        fold=fold, model=model, seed=seed,
                        mae=95.0, rmse=190.0, mape=28.0,
                        n_evaluated=1008, n_warmup_rows=0, n_ineligible_label_rows=6,
                    )
                )
    return records


def test_d_write_summary_on_synthetic_records(tmp_path):
    records = _synthetic_records()
    assert len(records) == 8 * (4 + 9)

    summary_path = tmp_path / "s.json"
    result = write_summary(
        records, summary_path,
        neural_models=["lstm", "gru", "cnn_lstm"],
        provenance={"note": "x"},
    )

    assert result["provenance"] == {"note": "x"}
    assert result["summary"]["n_records"] == 104
    assert result["summary"]["verdicts"]["lstm"]["linear_regression"]["verdict"] == "shown better"

    on_disk = json.loads(summary_path.read_text())
    assert on_disk == result

    with pytest.raises(FileExistsError):
        write_summary(
            records, summary_path,
            neural_models=["lstm", "gru", "cnn_lstm"],
            provenance={"note": "x"},
        )
    assert json.loads(summary_path.read_text()) == on_disk


def test_e_write_summary_refuses_before_computing(tmp_path):
    summary_path = tmp_path / "s.json"
    summary_path.write_text("junk")

    with pytest.raises(FileExistsError):
        write_summary(
            [], summary_path,
            neural_models=["lstm", "gru", "cnn_lstm"],
            provenance={"note": "x"},
        )
    assert summary_path.read_text() == "junk"


def test_f_main_end_to_end_deterministic_only(tmp_path):
    rc = main(["--out-dir", str(tmp_path), "--folds", "1", "--deterministic-only"])
    assert rc == 0

    records_path = tmp_path / RECORDS_NAME
    lines = records_path.read_text().splitlines()
    assert len(lines) == 4

    models = []
    for line in lines:
        row = json.loads(line)
        assert row["n_evaluated"] == 1008
        assert row["n_ineligible_label_rows"] == 6
        assert row["seed"] is None
        assert np.isfinite(row["mae"])
        assert np.isfinite(row["rmse"])
        assert np.isfinite(row["mape"])
        models.append(row["model"])

    assert models == ["naive_persistence", "naive_seasonal", "linear_regression", "random_forest"]
    assert not (tmp_path / SUMMARY_NAME).exists()


def test_g_main_refuses_early(tmp_path):
    records_path = tmp_path / RECORDS_NAME
    records_path.write_text("sentinel\n")

    with pytest.raises(FileExistsError):
        main(["--out-dir", str(tmp_path), "--folds", "1", "--deterministic-only"])
    assert records_path.read_text() == "sentinel\n"

    with pytest.raises((SystemExit, ValueError)):
        main(["--out-dir", str(tmp_path / "x"), "--folds", "9"])
