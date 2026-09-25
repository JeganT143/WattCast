"""Train and register a Phase 6 serving champion for a new model family,
generalizing scripts/train_final_lr.py — which is left in place, untouched,
and remains the script that produced the already-registered LR champion
(DECISIONS.md "Phase 6: serving registration, 2026-09-25" and "Phase 6:
per-fold selection evidence, 2026-09-25").

ALLOWED_FAMILIES is {"random_forest", "lstm", "gru", "cnn_lstm"}. linear_regression
is explicitly excluded — it is already registered via the original script,
and re-running it here would risk silently producing a second, divergent LR
run. The shared SequenceForecaster packaging/loading mechanism was verified
once against LSTM, then reapplied to gru and cnn_lstm (DECISIONS.md "Phase 6:
deep-model final deployment decision, 2026-09-25").
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from config.features import FEATURE_COLUMNS, SCALED_COLUMNS
from config.mlflow_config import TRACKING_URI
from config.paths import PROJECT_ROOT, RAW_DATA_PATH
from src.evaluation.walk_forward_models import HUBER_DELTA, LSTM_MAX_EPOCHS
from src.models.cnn_lstm import CNNLSTMForecaster
from src.models.gru import GRUForecaster
from src.models.lstm import LSTMForecaster
from src.models.sklearn_models import RandomForestForecaster
from src.serving.bundle import ModelBundle, feature_schema_version, save_bundle
from src.serving.bundle import required_raw_history as lr_rf_required_raw_history
from src.serving.buffer import SEQUENCE_MODEL_RAW_HISTORY
from src.serving.registry import CHAMPION_ALIAS, register_bundle, registered_model_name
from src.training.final_window import build_final_window

HORIZON = 6
SEED = 42  # DECISIONS.md "Phase 6: deep-model final deployment decision, 2026-09-25"
ALLOWED_FAMILIES = {"random_forest", "lstm", "gru", "cnn_lstm"}
SEQUENCE_FAMILIES = {"lstm", "gru", "cnn_lstm"}
SCALER_CONVENTION = (
    "StandardScaler fit on the untrimmed final-window feature frame "
    "(date < 2016-04-30), all rows True"
)
SEED_END = "2016-04-29 23:50:00"

_FORECASTER_FACTORIES = {
    "random_forest": RandomForestForecaster,
    # L, hidden_size, num_layers, dropout, learning_rate, batch_size,
    # torch_num_threads are deliberately NOT passed here: they have no
    # separate named constant anywhere in the repo for Phase 4/5 (the
    # walk-forward factories in src/evaluation/walk_forward_models.py
    # never override them either), so the existing SequenceForecaster
    # class defaults ARE the source of truth. Retyping them here would be
    # the config-drift risk DECISIONS.md warned against; omitting them
    # reuses the exact same defaults Phase 4/5 relied on. Only
    # max_epochs/huber_delta (LSTM_MAX_EPOCHS/HUBER_DELTA, imported above)
    # and seed (a fresh, distinct 42 for this final run) are provided.
    "lstm": lambda: LSTMForecaster(max_epochs=LSTM_MAX_EPOCHS, huber_delta=HUBER_DELTA, seed=SEED),
    "gru": lambda: GRUForecaster(max_epochs=LSTM_MAX_EPOCHS, huber_delta=HUBER_DELTA, seed=SEED),
    "cnn_lstm": lambda: CNNLSTMForecaster(
        max_epochs=LSTM_MAX_EPOCHS, huber_delta=HUBER_DELTA, seed=SEED
    ),
}


def _get_git_commit_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _results_path(family: str) -> Path:
    return PROJECT_ROOT / "results" / f"final_model_registration_{family}.json"


def check_can_run(family: str, results_path: Path | None, tracking_uri: str) -> None:
    if family == "linear_regression":
        raise ValueError(
            "linear_regression is already registered via scripts/train_final_lr.py; "
            "refusing to re-run it through this script"
        )
    if family not in ALLOWED_FAMILIES:
        raise ValueError(
            f"family={family!r} is not yet supported; allowed families are "
            f"{sorted(ALLOWED_FAMILIES)}"
        )

    if results_path is None:
        results_path = _results_path(family)
    if results_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing results file: {results_path}"
        )

    model_name = registered_model_name(family)
    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    except MlflowException:
        return
    raise RuntimeError(
        f"{model_name}@{CHAMPION_ALIAS} already exists; refusing to re-register"
    )


def main(
    family: str,
    results_path: Path | None = None,
    tracking_uri: str = TRACKING_URI,
    df_raw: pd.DataFrame | None = None,
) -> dict:
    if results_path is None:
        results_path = _results_path(family)

    check_can_run(family, results_path, tracking_uri)

    if df_raw is None:
        df_raw = pd.read_csv(RAW_DATA_PATH, parse_dates=["date"])
    window = build_final_window(df_raw, horizon=HORIZON)

    target_column = f"target_t{HORIZON}"
    fit_df = window.train_df.iloc[: window.n_fittable]

    forecaster = _FORECASTER_FACTORIES[family]()
    X = fit_df[forecaster.required_columns].to_numpy()
    y = fit_df[target_column].to_numpy()
    forecaster.fit(X, y)

    code_sha = _get_git_commit_sha()
    train_start = str(window.first_date)
    train_end = str(window.last_date)
    is_sequence_family = family in SEQUENCE_FAMILIES

    schema = {
        "model_family": family,
        "horizon": HORIZON,
        "feature_columns": FEATURE_COLUMNS,
        "scaled_columns": SCALED_COLUMNS,
        "required_columns": forecaster.required_columns,
        "required_raw_history": (
            SEQUENCE_MODEL_RAW_HISTORY if is_sequence_family else lr_rf_required_raw_history()
        ),
        "train_start": train_start,
        "train_end": train_end,
        "seed_end": SEED_END,
        "n_fit_rows": window.n_fittable,
        "scaler_convention": SCALER_CONVENTION,
        "code_sha": code_sha,
        "feature_schema_version": feature_schema_version(FEATURE_COLUMNS),
    }

    if is_sequence_family:
        # Sourced entirely from the fitted instance itself (forecaster.params,
        # the existing SequenceForecaster source of truth for its own
        # constructor args) — never retyped literals, so this can't drift
        # from what was actually built and trained.
        schema["architecture"] = {
            "class": type(forecaster._net).__name__,
            "n_features": len(forecaster.required_columns),
            "params": forecaster.params,
        }
        schema["seed"] = forecaster.seed
        schema["epochs"] = forecaster.max_epochs

    bundle = ModelBundle(forecaster=forecaster, scaler=window.scaler, schema=schema)

    with tempfile.TemporaryDirectory() as tmp_dir:
        bundle_dir = Path(tmp_dir) / "bundle"
        save_bundle(bundle, bundle_dir)

        tags = {
            "train_start": train_start,
            "train_end": train_end,
            "model_family": family,
            "horizon": HORIZON,
            "feature_schema_version": schema["feature_schema_version"],
            "scaler_convention": SCALER_CONVENTION,
            "code_sha": code_sha,
            "selection_basis": (
                "post-hoc; DECISIONS.md Phase 6 registration 45788c2 "
                "and per-fold note 30ccfd3"
            ),
            "test_partition_used": "false",
        }
        if is_sequence_family:
            tags["seed"] = schema["seed"]
            tags["epochs"] = schema["epochs"]
        run_id, model_version = register_bundle(
            bundle_dir, tracking_uri, tags=tags, family=family
        )

    tracking_store_path = Path(tracking_uri.removeprefix("sqlite:///"))
    try:
        tracking_store_relative = str(tracking_store_path.relative_to(PROJECT_ROOT))
    except ValueError:
        tracking_store_relative = str(tracking_store_path)

    payload = {
        "run_id": run_id,
        "model_version": model_version,
        "model_family": family,
        "n_mask_rows": window.n_mask_rows,
        "n_lag144_finite": window.n_lag144_finite,
        "n_fittable": window.n_fittable,
        "code_sha": code_sha,
        "train_start": train_start,
        "train_end": train_end,
        "tracking_store_path": tracking_store_relative,
    }
    if is_sequence_family:
        payload["seed"] = schema["seed"]
        payload["epochs"] = schema["epochs"]

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    args = parser.parse_args()
    result = main(family=args.family)
    print(json.dumps(result, indent=2, sort_keys=True))
