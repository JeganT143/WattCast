"""Train and register a Phase 6 serving champion for a new model family,
generalizing scripts/train_final_lr.py — which is left in place, untouched,
and remains the script that produced the already-registered LR champion
(DECISIONS.md "Phase 6: serving registration, 2026-09-25" and "Phase 6:
per-fold selection evidence, 2026-09-25").

ALLOWED_FAMILIES is exactly {"random_forest"} in this stage. linear_regression
is explicitly excluded — it is already registered via the original script,
and re-running it here would risk silently producing a second, divergent LR
run. Deep sequence-model families (lstm/gru/cnn_lstm) are a later stage.
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
from src.models.sklearn_models import RandomForestForecaster
from src.serving.bundle import ModelBundle, feature_schema_version, required_raw_history, save_bundle
from src.serving.registry import CHAMPION_ALIAS, register_bundle, registered_model_name
from src.training.final_window import build_final_window

HORIZON = 6
ALLOWED_FAMILIES = {"random_forest"}
SCALER_CONVENTION = (
    "StandardScaler fit on the untrimmed final-window feature frame "
    "(date < 2016-04-30), all rows True"
)
SEED_END = "2016-04-29 23:50:00"

_FORECASTER_FACTORIES = {
    "random_forest": RandomForestForecaster,
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

    schema = {
        "model_family": family,
        "horizon": HORIZON,
        "feature_columns": FEATURE_COLUMNS,
        "scaled_columns": SCALED_COLUMNS,
        "required_columns": forecaster.required_columns,
        "required_raw_history": required_raw_history(),
        "train_start": train_start,
        "train_end": train_end,
        "seed_end": SEED_END,
        "n_fit_rows": window.n_fittable,
        "scaler_convention": SCALER_CONVENTION,
        "code_sha": code_sha,
        "feature_schema_version": feature_schema_version(FEATURE_COLUMNS),
    }
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
