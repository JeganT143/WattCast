"""Golden LSTM fixture: a small, validation-only freeze of LSTMForecaster's
behaviour, generated once BEFORE the SequenceForecaster refactor begins.

Run from the repo root:  PYTHONPATH=. python tests/golden_lstm.py
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch

from src.evaluation.validation_only import evaluate_on_validation
from src.models.lstm import LSTMForecaster

TRAIN_PATH = "data/processed/train_t6.csv"
VAL_PATH = "data/processed/val_t6.csv"
FIXTURE_PATH = "tests/fixtures/golden_lstm.json"
LSTM_PY_PATH = "src/models/lstm.py"

CONFIGS = (
    {"seed": 42, "max_epochs": 2, "huber_delta": 40.0},
    {"seed": 43, "max_epochs": 2, "huber_delta": 40.0},
)


def fit_and_summarize(config: dict, **model_overrides) -> dict:
    train_df = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    val_df = pd.read_csv(VAL_PATH, parse_dates=["date"])

    forecaster = LSTMForecaster(**config, **model_overrides)
    result = evaluate_on_validation(forecaster, train_df, val_df, "target_t6")

    predictions_sha256 = hashlib.sha256(
        np.ascontiguousarray(result.predictions, dtype=np.float64).tobytes()
    ).hexdigest()

    return {
        "config": config,
        "params": forecaster.params,
        "mae": float(result.metrics["mae"]),
        "rmse": float(result.metrics["rmse"]),
        "n_evaluated": int(result.n_evaluated),
        "n_warmup_rows": int(result.n_warmup_rows),
        "n_ineligible_label_rows": int(result.n_ineligible_label_rows),
        "training_losses": [float(x) for x in forecaster.training_losses_],
        "initial_output_bias": float(forecaster.initial_output_bias_),
        "predictions_head": [float(x) for x in result.predictions[:5]],
        "predictions_tail": [float(x) for x in result.predictions[-5:]],
        "predictions_sha256": predictions_sha256,
    }


def build_golden() -> dict:
    lstm_py_sha256 = hashlib.sha256(open(LSTM_PY_PATH, "rb").read()).hexdigest()
    return {
        "provenance": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_num_threads": 4,
            "lstm_py_sha256": lstm_py_sha256,
            "note": (
                "generated once before the SequenceForecaster refactor; "
                "never regenerate after the refactor begins"
            ),
        },
        "runs": [fit_and_summarize(c) for c in CONFIGS],
    }


def write_golden(path=FIXTURE_PATH):
    if os.path.exists(path):
        raise FileExistsError(f"{path} already exists; refusing to overwrite the golden fixture")
    golden = build_golden()
    with open(path, "w") as f:
        f.write(json.dumps(golden, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path


if __name__ == "__main__":
    written_path = write_golden()
    print(written_path)
