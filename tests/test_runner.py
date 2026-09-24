import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.features import FEATURE_COLUMNS
from src.tuning.runner import main, run_grid
from src.tuning.selection import select_configuration

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "tuning" / "runner.py"
TARGET = "target_t6"
COLUMNS = [c for c in FEATURE_COLUMNS if c not in {"hour_of_day", "day_of_week"}]
SMALL = dict(L=4, hidden_size=8, batch_size=16, torch_num_threads=1)
EPOCHS, DELTAS, SEEDS = (1, 2), (20.0, 40.0), (0, 1)
REF_MAE, REF_RMSE = 40.0, 80.0


def _inputs(tmp_path):
    rng = np.random.default_rng(0)
    n_train, n_val = 60, 20
    n = n_train + n_val
    df = pd.DataFrame(rng.normal(size=(n, len(COLUMNS))), columns=COLUMNS)
    df.insert(0, "date", pd.date_range("2016-01-01", periods=n, freq="10min"))
    df[TARGET] = 60.0 + 10.0 * df[COLUMNS[0]] + rng.normal(scale=2.0, size=n)
    train = df.iloc[:n_train].copy()
    train.loc[train.index[-2:], TARGET] = np.nan
    val = df.iloc[n_train:]
    paths = {
        "train_path": tmp_path / "train.csv",
        "val_path": tmp_path / "val.csv",
        "reference_path": tmp_path / "golden.json",
    }
    train.to_csv(paths["train_path"], index=False)
    val.to_csv(paths["val_path"], index=False)
    paths["reference_path"].write_text(
        json.dumps(
            {
                "meta": {},
                "results": {
                    "linear_regression|h6": {
                        "metrics": {
                            "val_mae": REF_MAE,
                            "val_rmse": REF_RMSE,
                            "test_mae": 8765.4321,
                            "test_rmse": 4321.8765,
                        }
                    }
                },
            }
        )
    )
    return paths


def _run(tmp_path, out_name="out.json", **overrides):
    kwargs = dict(seeds=SEEDS, epochs=EPOCHS, deltas=DELTAS, model_kwargs=SMALL)
    kwargs.update(overrides)
    return run_grid(**_inputs_cached(tmp_path), out_path=tmp_path / out_name, **kwargs)


def _inputs_cached(tmp_path):
    if not (tmp_path / "train.csv").exists():
        _inputs(tmp_path)
    return {
        "train_path": tmp_path / "train.csv",
        "val_path": tmp_path / "val.csv",
        "reference_path": tmp_path / "golden.json",
    }


def test_every_combination_is_fitted_once_and_reported(tmp_path):
    out = _run(tmp_path)
    assert set(out) == {"protocol", "reference", "runs", "configurations", "selected"}
    assert len(out["runs"]) == 8
    combos = {(r["max_epochs"], r["huber_delta"], r["seed"]) for r in out["runs"]}
    assert combos == {(e, d, s) for e in EPOCHS for d in DELTAS for s in SEEDS}
    for r in out["runs"]:
        assert r["n_evaluated"] == 20 and r["n_warmup_rows"] == 3 and r["n_ineligible_label_rows"] == 2
        assert np.isfinite(r["val_mae"]) and np.isfinite(r["val_rmse"])
    assert len(out["configurations"]) == 4
    assert out["reference"] == {"lr_val_mae": REF_MAE, "lr_val_rmse": REF_RMSE}
    assert out["protocol"]["seeds"] == list(SEEDS)
    assert out["protocol"]["epochs"] == list(EPOCHS)
    assert out["protocol"]["deltas"] == list(DELTAS)
    assert out["protocol"]["target_column"] == TARGET
    assert out == json.loads((tmp_path / "out.json").read_text())


def test_configuration_metrics_are_the_seed_means_and_ratios_use_the_reference(tmp_path):
    out = _run(tmp_path)
    for c in out["configurations"]:
        runs = [r for r in out["runs"] if (r["max_epochs"], r["huber_delta"]) == (c["max_epochs"], c["huber_delta"])]
        assert len(runs) == len(SEEDS)
        assert c["mean_val_mae"] == pytest.approx(np.mean([r["val_mae"] for r in runs]), rel=1e-12)
        assert c["mean_val_rmse"] == pytest.approx(np.mean([r["val_rmse"] for r in runs]), rel=1e-12)
        assert c["val_mae_ratio"] == pytest.approx(c["mean_val_mae"] / REF_MAE, rel=1e-12)
        assert c["val_rmse_ratio"] == pytest.approx(c["mean_val_rmse"] / REF_RMSE, rel=1e-12)
        assert c["selection_score"] == max(c["val_mae_ratio"], c["val_rmse_ratio"])


def test_selected_configuration_is_the_selection_rule_applied_to_the_scored_rows(tmp_path):
    out = _run(tmp_path)
    rows = [
        {k: c[k] for k in ("max_epochs", "huber_delta", "mean_val_mae", "mean_val_rmse")}
        for c in out["configurations"]
    ]
    expected = select_configuration(rows, REF_MAE, REF_RMSE)
    assert (out["selected"]["max_epochs"], out["selected"]["huber_delta"]) == (
        expected["max_epochs"],
        expected["huber_delta"],
    )
    assert out["selected"]["selection_score"] == min(c["selection_score"] for c in out["configurations"])


def test_runs_with_only_the_train_validation_and_reference_files_present(tmp_path):
    _inputs(tmp_path)
    before = {p.name for p in tmp_path.iterdir()}
    assert before == {"train.csv", "val.csv", "golden.json"}
    run_grid(**_inputs_cached(tmp_path), out_path=tmp_path / "out.json", seeds=SEEDS, epochs=EPOCHS, deltas=DELTAS, model_kwargs=SMALL)
    assert {p.name for p in tmp_path.iterdir()} == before | {"out.json"}


def test_reads_exactly_the_three_input_files(tmp_path, monkeypatch):
    paths = _inputs_cached(tmp_path)
    seen = []
    real_csv, real_text = pd.read_csv, Path.read_text

    def spy_csv(path, *args, **kwargs):
        seen.append(str(path))
        return real_csv(path, *args, **kwargs)

    def spy_text(self, *args, **kwargs):
        seen.append(str(self))
        return real_text(self, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy_csv)
    monkeypatch.setattr(Path, "read_text", spy_text)
    run_grid(**paths, out_path=tmp_path / "out.json", seeds=SEEDS, epochs=EPOCHS, deltas=DELTAS, model_kwargs=SMALL)
    assert sorted(set(seen)) == sorted(str(p) for p in paths.values())


def test_output_carries_no_value_from_the_reference_test_metrics(tmp_path):
    _run(tmp_path)
    text = (tmp_path / "out.json").read_text()
    assert "8765.4321" not in text and "4321.8765" not in text


def test_output_is_deterministic(tmp_path):
    _run(tmp_path, "a.json")
    _run(tmp_path, "b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


@pytest.mark.parametrize("name", ["max_epochs", "huber_delta", "seed"])
def test_model_kwargs_cannot_override_grid_parameters(tmp_path, name):
    with pytest.raises(ValueError, match="grid"):
        _run(tmp_path, model_kwargs={**SMALL, name: 1})


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
    ],
)
def test_grep_gate_forbidden_pattern_is_absent(name, pattern):
    hits = [(i, line) for i, line in enumerate(_source().splitlines(), 1) if re.search(pattern, line)]
    assert not hits, (name, hits)


def test_module_uses_the_validation_only_evaluator_and_the_selection_module():
    src = _source()
    assert re.search(r"from src\.evaluation\.validation_only import", src)
    assert re.search(r"from src\.tuning\.selection import", src)
    assert re.search(r"from src\.models\.lstm import", src)


def test_command_line_entry_point_writes_the_output(tmp_path, capsys):
    paths = _inputs_cached(tmp_path)
    out_path = tmp_path / "cli.json"
    code = main(
        [
            "--train", str(paths["train_path"]),
            "--val", str(paths["val_path"]),
            "--reference", str(paths["reference_path"]),
            "--out", str(out_path),
            "--target", TARGET,
            "--epochs", "1", "2",
            "--deltas", "20", "40",
            "--seeds", "0", "1",
        ]
    )
    assert code == 0
    data = json.loads(out_path.read_text())
    assert len(data["configurations"]) == 4 and len(data["runs"]) == 8
    assert "selected" in capsys.readouterr().out
