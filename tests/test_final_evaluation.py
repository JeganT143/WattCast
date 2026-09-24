import json
import math
import re
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.features import FEATURE_COLUMNS
from src.evaluation.final_evaluation import (
    REGISTERED_HORIZONS,
    REGISTERED_SEEDS,
    TIER1_FACTOR,
    load_selected_configuration,
    load_tier1_reference,
    main,
    run_final_evaluation,
    summarize,
    tier1_verdict,
)
from src.models.lstm import LSTMForecaster

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "evaluation" / "final_evaluation.py"
COLUMNS = [c for c in FEATURE_COLUMNS if c not in {"hour_of_day", "day_of_week"}]
MODELS = ("naive_persistence", "naive_seasonal", "linear_regression", "random_forest")
REF_MAE, REF_RMSE = 40.0, 80.0
T_MAE, T_RMSE = 0.99 * REF_MAE, 0.99 * REF_RMSE
N_TRAIN, N_VAL, N_TEST = 60, 20, 15


def _results(maes, rmses, seeds=(42, 43, 44)):
    return [{"seed": s, "test_mae": m, "test_rmse": r} for s, m, r in zip(seeds, maes, rmses)]


def _write_fixture(path, lr_h6_test_mae=42.391318123456789, lr_h6_test_rmse=80.619123456789012):
    results = {}
    for mi, model in enumerate(MODELS):
        for h in (1, 6):
            base = 10.0 * (mi + 1) + h
            metrics = {}
            for pi, part in enumerate(("train", "val", "test")):
                for ki, kind in enumerate(("mae", "rmse", "mape")):
                    metrics[f"{part}_{kind}"] = base + 100.0 * pi + 0.25 * ki
            results[f"{model}|h{h}"] = {"metrics": metrics, "n_rows": {}, "n_evaluated": {}}
    results["linear_regression|h6"]["metrics"]["test_mae"] = lr_h6_test_mae
    results["linear_regression|h6"]["metrics"]["test_rmse"] = lr_h6_test_rmse
    path.write_text(json.dumps({"meta": {}, "results": results}))


def _write_tuning(path, max_epochs=2, huber_delta=60.0, **overrides):
    params = {
        "L": 4, "batch_size": 16, "dropout": 0.0, "hidden_size": 8, "learning_rate": 0.001,
        "loss": "huber", "num_layers": 1, "optimizer": "adam",
        "output_bias_init": "train_target_median", "torch_num_threads": 1,
    }
    params.update(overrides)
    doc = {
        "protocol": {"model_params": params, "seeds": [42, 43, 44]},
        "selected": {
            "max_epochs": max_epochs, "huber_delta": huber_delta,
            "selection_score": 0.5, "val_mae_ratio": 0.4, "val_rmse_ratio": 0.5,
        },
        "runs": [], "configurations": [], "reference": {},
    }
    path.write_text(json.dumps(doc))
    return params


def _write_data(data_dir):
    rng = np.random.default_rng(0)
    n = N_TRAIN + N_VAL + N_TEST
    df = pd.DataFrame(rng.normal(size=(n, len(COLUMNS))), columns=COLUMNS)
    df.insert(0, "date", pd.date_range("2016-01-01", periods=n, freq="10min"))
    for h in REGISTERED_HORIZONS:
        df[f"target_t{h}"] = 60.0 + 10.0 * df[COLUMNS[0]] + rng.normal(scale=2.0, size=n) + h
    for h in REGISTERED_HORIZONS:
        part = df[["date", *COLUMNS, f"target_t{h}"]]
        train = part.iloc[:N_TRAIN].copy()
        train.loc[train.index[-h:], f"target_t{h}"] = np.nan
        train.to_csv(data_dir / f"train_t{h}.csv", index=False)
        part.iloc[N_TRAIN : N_TRAIN + N_VAL].to_csv(data_dir / f"val_t{h}.csv", index=False)
        part.iloc[N_TRAIN + N_VAL :].to_csv(data_dir / f"test_t{h}.csv", index=False)
    return df.iloc[N_TRAIN + N_VAL :]["target_t6"].to_numpy()


def _setup(tmp_path, ref_scale):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    actual6 = _write_data(data_dir)
    tuning = tmp_path / "tuning.json"
    params = _write_tuning(tuning)
    fixture = tmp_path / "golden.json"
    _write_fixture(fixture, 40.0 * ref_scale, 80.0 * ref_scale)
    paths = dict(
        data_dir=data_dir,
        tuning_results_path=tuning,
        reference_path=fixture,
        out_path=tmp_path / "final.json",
        predictions_path=tmp_path / "preds.csv",
    )
    return paths, actual6, params


# ---------------------------------------------------------------------
# Registered constants and the Tier 1 verdict
# ---------------------------------------------------------------------


def test_registered_constants():
    assert REGISTERED_SEEDS == (42, 43, 44)
    assert REGISTERED_HORIZONS == (6, 1)
    assert TIER1_FACTOR == 0.99


def test_tier1_all_seeds_passing_is_shown():
    v = tier1_verdict(_results([30.0, 31.0, 32.0], [70.0, 71.0, 72.0]), REF_MAE, REF_RMSE)
    assert v["shown"] is True
    assert v["factor"] == 0.99
    assert v["threshold_mae"] == T_MAE and v["threshold_rmse"] == T_RMSE
    assert [p["seed"] for p in v["per_seed"]] == [42, 43, 44]
    assert all(p["mae_pass"] and p["rmse_pass"] and p["pass"] for p in v["per_seed"])


def test_tier1_one_seed_missing_the_rmse_threshold_is_not_shown():
    v = tier1_verdict(_results([30.0, 31.0, 32.0], [70.0, 81.0, 72.0]), REF_MAE, REF_RMSE)
    assert v["shown"] is False
    assert [p["rmse_pass"] for p in v["per_seed"]] == [True, False, True]
    assert [p["mae_pass"] for p in v["per_seed"]] == [True, True, True]
    assert [p["pass"] for p in v["per_seed"]] == [True, False, True]


def test_tier1_needs_both_metrics_whichever_one_fails():
    rmse_fails = tier1_verdict(_results([30.0] * 3, [85.0] * 3), REF_MAE, REF_RMSE)
    assert rmse_fails["shown"] is False
    assert all(p["mae_pass"] for p in rmse_fails["per_seed"])
    assert not any(p["rmse_pass"] for p in rmse_fails["per_seed"])
    assert not any(p["pass"] for p in rmse_fails["per_seed"])
    mae_fails = tier1_verdict(_results([45.0] * 3, [70.0] * 3), REF_MAE, REF_RMSE)
    assert mae_fails["shown"] is False
    assert not any(p["mae_pass"] for p in mae_fails["per_seed"])
    assert all(p["rmse_pass"] for p in mae_fails["per_seed"])
    assert not any(p["pass"] for p in mae_fails["per_seed"])


def test_tier1_exact_threshold_equality_passes_and_one_ulp_above_fails():
    at = tier1_verdict(_results([T_MAE] * 3, [T_RMSE] * 3), REF_MAE, REF_RMSE)
    assert at["shown"] is True
    above_mae = math.nextafter(T_MAE, math.inf)
    v = tier1_verdict(_results([T_MAE, above_mae, T_MAE], [T_RMSE] * 3), REF_MAE, REF_RMSE)
    assert [p["mae_pass"] for p in v["per_seed"]] == [True, False, True]
    assert v["shown"] is False
    above_rmse = math.nextafter(T_RMSE, math.inf)
    v = tier1_verdict(_results([T_MAE] * 3, [T_RMSE, T_RMSE, above_rmse]), REF_MAE, REF_RMSE)
    assert [p["rmse_pass"] for p in v["per_seed"]] == [True, True, False]
    assert v["shown"] is False


def test_tier1_all_seeds_failing_is_not_shown():
    v = tier1_verdict(_results([50.0, 51.0, 52.0], [90.0, 91.0, 92.0]), REF_MAE, REF_RMSE)
    assert v["shown"] is False
    assert not any(p["mae_pass"] or p["rmse_pass"] or p["pass"] for p in v["per_seed"])


def test_tier1_mean_performance_cannot_rescue_a_failing_seed():
    v = tier1_verdict(_results([1.0, 1.0, 100.0], [10.0] * 3), REF_MAE, REF_RMSE)
    assert sum([1.0, 1.0, 100.0]) / 3 < T_MAE
    assert v["shown"] is False
    assert [p["pass"] for p in v["per_seed"]] == [True, True, False]


@pytest.mark.parametrize("seeds", [(42, 43), (42, 43, 44, 45), (1, 2, 3)])
def test_tier1_requires_exactly_the_registered_seeds(seeds):
    with pytest.raises(ValueError, match="seeds"):
        tier1_verdict(_results([30.0] * len(seeds), [70.0] * len(seeds), seeds=seeds), REF_MAE, REF_RMSE)


# ---------------------------------------------------------------------
# Reference thresholds, selected configuration, summary statistics
# ---------------------------------------------------------------------


def test_thresholds_are_derived_from_the_unrounded_h6_test_reference(tmp_path):
    path = tmp_path / "golden.json"
    _write_fixture(path)
    ref = load_tier1_reference(path)
    assert set(ref) == {"ref_mae", "ref_rmse", "threshold_mae", "threshold_rmse"}
    assert ref["ref_mae"] == 42.391318123456789
    assert ref["ref_rmse"] == 80.619123456789012
    assert ref["threshold_mae"] == 0.99 * 42.391318123456789
    assert ref["threshold_rmse"] == 0.99 * 80.619123456789012
    entries = json.loads(path.read_text())["results"]
    assert ref["ref_mae"] != entries["linear_regression|h6"]["metrics"]["val_mae"]
    assert ref["ref_mae"] != entries["linear_regression|h1"]["metrics"]["test_mae"]
    assert ref["ref_mae"] != entries["random_forest|h6"]["metrics"]["test_mae"]


def test_missing_reference_entry_raises(tmp_path):
    path = tmp_path / "golden.json"
    path.write_text(json.dumps({"meta": {}, "results": {}}))
    with pytest.raises(ValueError, match="linear_regression"):
        load_tier1_reference(path)


def test_selected_configuration_is_read_from_the_results_structure_not_hard_coded(tmp_path):
    path = tmp_path / "tuning.json"
    params = _write_tuning(path, max_epochs=35, huber_delta=20.0, L=6, hidden_size=12, batch_size=32)
    cfg = load_selected_configuration(path)
    assert cfg == {"max_epochs": 35, "huber_delta": 20.0, "model_kwargs": params}
    assert cfg["model_kwargs"]["L"] == 6 and cfg["model_kwargs"]["hidden_size"] == 12


def test_the_committed_tuning_result_is_read_as_is():
    path = ROOT / "results" / "tuning_h6_validation.json"
    if not path.exists():
        pytest.skip("committed tuning result not present")
    data = json.loads(path.read_text())
    cfg = load_selected_configuration(path)
    assert set(cfg) == {"max_epochs", "huber_delta", "model_kwargs"}
    assert (cfg["max_epochs"], cfg["huber_delta"]) == (data["selected"]["max_epochs"], data["selected"]["huber_delta"])
    assert cfg["model_kwargs"] == data["protocol"]["model_params"]


def test_summarize_uses_the_sample_standard_deviation_with_ddof_1():
    values = [1.0, 2.0, 4.0]
    s = summarize(values)
    assert set(s) == {"mean", "std"}
    assert s["mean"] == pytest.approx(7.0 / 3.0, rel=1e-12)
    assert s["std"] == pytest.approx(math.sqrt(7.0 / 3.0), rel=1e-12)
    assert s["std"] == pytest.approx(statistics.stdev(values), rel=1e-12)
    assert s["std"] != pytest.approx(statistics.pstdev(values), rel=1e-6)


def test_summarize_needs_at_least_two_values():
    with pytest.raises(ValueError, match="at least two"):
        summarize([1.0])


# ---------------------------------------------------------------------
# Running the evaluation
# ---------------------------------------------------------------------


@pytest.mark.parametrize("which", ["out", "predictions"])
def test_refuses_to_overwrite_existing_outputs_before_reading_anything(tmp_path, which):
    out, preds = tmp_path / "final.json", tmp_path / "preds.csv"
    existing, other = (out, preds) if which == "out" else (preds, out)
    existing.write_text("sentinel")
    with pytest.raises(FileExistsError):
        run_final_evaluation(
            data_dir=tmp_path / "does_not_exist",
            tuning_results_path=tmp_path / "missing_tuning.json",
            reference_path=tmp_path / "missing_reference.json",
            out_path=out,
            predictions_path=preds,
        )
    assert existing.read_text() == "sentinel"
    assert not other.exists()


@pytest.mark.parametrize("ref_scale", [1.0e6, 1.0e-6])
def test_tiny_synthetic_end_to_end_reports_every_seed_and_horizon_whatever_the_outcome(tmp_path, monkeypatch, ref_scale):
    paths, actual6, params = _setup(tmp_path, ref_scale)
    fitted = []
    real_fit = LSTMForecaster.fit

    def counting_fit(self, X, y):
        fitted.append(self.params["seed"])
        return real_fit(self, X, y)

    monkeypatch.setattr(LSTMForecaster, "fit", counting_fit)
    result = run_final_evaluation(**paths)

    assert sorted(fitted) == [42, 42, 43, 43, 44, 44]
    assert result == json.loads(paths["out_path"].read_text())
    assert result["protocol"]["seeds"] == [42, 43, 44]
    assert result["protocol"]["horizons"] == [6, 1]
    assert set(result["horizons"]) == {"6", "1"}
    for h in ("6", "1"):
        runs = result["horizons"][h]["runs"]
        assert [r["seed"] for r in runs] == [42, 43, 44]
        for r in runs:
            assert r["params"]["max_epochs"] == 2 and r["params"]["huber_delta"] == 60.0
            assert r["params"]["L"] == params["L"] and r["params"]["hidden_size"] == params["hidden_size"]
            assert r["params"]["seed"] == r["seed"]
            assert np.isfinite(r["test_mae"]) and np.isfinite(r["test_rmse"]) and np.isfinite(r["test_mape"])
            assert r["n_evaluated"]["test"] == N_TEST
            assert r["n_warmup_rows"] == params["L"] - 1
            assert r["n_ineligible_label_rows"] == int(h)
        for metric in ("test_mae", "test_rmse"):
            values = [r[metric] for r in runs]
            summary = result["horizons"][h]["summary"][metric]
            assert summary["mean"] == pytest.approx(statistics.fmean(values), rel=1e-12)
            assert summary["std"] == pytest.approx(statistics.stdev(values), rel=1e-12)

    ref = load_tier1_reference(paths["reference_path"])
    assert result["tier1_reference"] == ref
    runs6 = result["horizons"]["6"]["runs"]
    verdict = tier1_verdict(
        [{"seed": r["seed"], "test_mae": r["test_mae"], "test_rmse": r["test_rmse"]} for r in runs6],
        ref["ref_mae"],
        ref["ref_rmse"],
    )
    assert result["tier1"] == verdict
    assert [p["seed"] for p in result["tier1"]["per_seed"]] == [42, 43, 44]
    assert result["tier1"]["shown"] == (ref_scale > 1.0)

    fixture = json.loads(paths["reference_path"].read_text())["results"]
    assert set(result["baselines"]) == set(fixture)
    for key, entry in fixture.items():
        for metric in ("test_mae", "test_rmse", "test_mape"):
            assert result["baselines"][key][metric] == entry["metrics"][metric]

    preds = pd.read_csv(paths["predictions_path"], parse_dates=["date"])
    assert list(preds.columns) == ["date", "actual", "pred_seed_42", "pred_seed_43", "pred_seed_44"]
    assert len(preds) == N_TEST
    expected_dates = pd.date_range("2016-01-01", periods=N_TRAIN + N_VAL + N_TEST, freq="10min")[N_TRAIN + N_VAL :]
    assert list(preds["date"]) == list(expected_dates)
    assert np.allclose(preds["actual"].to_numpy(), actual6, rtol=1e-12)
    assert np.isfinite(preds[["pred_seed_42", "pred_seed_43", "pred_seed_44"]].to_numpy()).all()


def test_command_line_entry_point_returns_zero_even_when_tier1_is_not_shown(tmp_path, capsys):
    paths, _, _ = _setup(tmp_path, 1.0e-6)
    code = main(
        [
            "--data-dir", str(paths["data_dir"]),
            "--tuning-results", str(paths["tuning_results_path"]),
            "--reference", str(paths["reference_path"]),
            "--out", str(paths["out_path"]),
            "--predictions", str(paths["predictions_path"]),
        ]
    )
    assert code == 0
    assert json.loads(paths["out_path"].read_text())["tier1"]["shown"] is False
    assert "tier 1" in capsys.readouterr().out.lower()


def test_module_contains_no_selection_or_tuning_logic():
    src = MODULE.read_text(encoding="utf-8")
    assert not re.search(r"from src\.tuning|import src\.tuning|select_configuration|score_configurations|run_grid", src)
    assert re.search(r"from src\.evaluation\.harness import", src)
