"""Train and register the Phase 6 serving champion: linear regression at
h=6, fit on the final train+validation window (decisions.md, ADR-012 and ADR-013).

Thin orchestration only — the window, model, bundle, and registry pieces
are all implemented elsewhere. Refuses to run twice: once results/
final_lr_registration.json exists, or once the registered model already
has a champion alias, this script stops rather than silently
re-registering a new champion version.
"""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from config.features import FEATURE_COLUMNS, SCALED_COLUMNS
from config.mlflow_config import TRACKING_URI
from config.paths import PROJECT_ROOT, RAW_DATA_PATH
from src.serving.bundle import ModelBundle, feature_schema_version, required_raw_history, save_bundle
from src.serving.registry import CHAMPION_ALIAS, REGISTERED_MODEL_NAME, register_bundle
from src.models.sklearn_models import LinearRegressionForecaster
from src.training.final_window import build_final_window

HORIZON = 6
RESULTS_PATH = PROJECT_ROOT / "results" / "final_lr_registration.json"
SCALER_CONVENTION = (
    "StandardScaler fit on the untrimmed final-window feature frame "
    "(date < 2016-04-30), all rows True"
)
SEED_END = "2016-04-29 23:50:00"


def _get_git_commit_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def check_can_run(results_path: Path, tracking_uri: str) -> None:
    if results_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing results file: {results_path}"
        )
    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    except MlflowException:
        return
    raise RuntimeError(
        f"{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS} already exists; refusing to re-register"
    )


def main(
    results_path: Path = RESULTS_PATH, tracking_uri: str = TRACKING_URI
) -> dict:
    check_can_run(results_path, tracking_uri)

    df_raw = pd.read_csv(RAW_DATA_PATH, parse_dates=["date"])
    window = build_final_window(df_raw, horizon=HORIZON)

    target_column = f"target_t{HORIZON}"
    fit_df = window.train_df.iloc[: window.n_fittable]
    X = fit_df[FEATURE_COLUMNS].to_numpy()
    y = fit_df[target_column].to_numpy()

    forecaster = LinearRegressionForecaster()
    forecaster.fit(X, y)

    code_sha = _get_git_commit_sha()
    train_start = str(window.first_date)
    train_end = str(window.last_date)

    schema = {
        "model_family": "linear_regression",
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
            "model_family": schema["model_family"],
            "horizon": HORIZON,
            "feature_schema_version": schema["feature_schema_version"],
            "scaler_convention": SCALER_CONVENTION,
            "code_sha": code_sha,
            "selection_basis": (
                "post-hoc; decisions.md ADR-012 (research log commits 45788c2, 30ccfd3)"
            ),
            "test_partition_used": "false",
        }
        run_id, model_version = register_bundle(bundle_dir, tracking_uri, tags=tags)

    coef = np.asarray(forecaster.model.coef_, dtype=np.float64)
    intercept = float(forecaster.model.intercept_)
    coef_sha256 = hashlib.sha256(
        coef.tobytes() + np.float64(intercept).tobytes()
    ).hexdigest()

    tracking_store_path = tracking_uri.removeprefix("sqlite:///")
    tracking_store_relative = str(Path(tracking_store_path).relative_to(PROJECT_ROOT))

    payload = {
        "run_id": run_id,
        "model_version": model_version,
        "n_mask_rows": window.n_mask_rows,
        "n_lag144_finite": window.n_lag144_finite,
        "n_fittable": window.n_fittable,
        "coef_sha256": coef_sha256,
        "intercept": intercept,
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
    result = main()
    print(json.dumps(result, indent=2, sort_keys=True))
