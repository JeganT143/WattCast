"""Tests for the frozen validation-only LSTM golden (tests/golden_lstm.py),
generated once before the SequenceForecaster refactor of src/models/lstm.py
begins. See tests/test_golden_baselines.py for the equivalent pattern used
by the baseline models."""

import json

import pytest
import torch

from tests.golden_lstm import CONFIGS, FIXTURE_PATH, fit_and_summarize, write_golden


def _load_fixture():
    with open(FIXTURE_PATH) as f:
        return json.load(f)


def _fixture_or_skip():
    fixture = _load_fixture()
    fixture_torch = fixture["provenance"]["torch"]
    if torch.__version__ != fixture_torch:
        pytest.skip(f"torch version mismatch: fixture={fixture_torch} runtime={torch.__version__}")
    return fixture


@pytest.mark.requires_processed_data
def test_a_exact_reproduction():
    fixture = _fixture_or_skip()
    for config, expected_run in zip(CONFIGS, fixture["runs"]):
        fresh = fit_and_summarize(config)
        assert fresh == expected_run


def test_b_runs_differ_by_seed():
    fixture = _fixture_or_skip()
    run0, run1 = fixture["runs"]
    assert run0["mae"] != run1["mae"]
    assert run0["rmse"] != run1["rmse"]
    assert run0["predictions_sha256"] != run1["predictions_sha256"]


@pytest.mark.requires_processed_data
def test_c_perturbation_detected():
    fixture = _fixture_or_skip()
    baseline = fixture["runs"][0]
    perturbed = fit_and_summarize(CONFIGS[0], learning_rate=1.1e-3)
    assert perturbed["mae"] != baseline["mae"]
    assert perturbed["training_losses"] != baseline["training_losses"]
    assert perturbed["predictions_sha256"] != baseline["predictions_sha256"]


def test_d_write_golden_refuses_existing_file(tmp_path):
    target = tmp_path / "g.json"
    target.write_text("sentinel")

    with pytest.raises(FileExistsError):
        write_golden(target)

    assert target.read_text() == "sentinel"
