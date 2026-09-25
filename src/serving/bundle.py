"""Serving model bundle: a fitted forecaster, its scaler, and the schema
that pins the feature contract they were trained against
(DECISIONS.md "Phase 6: serving registration, 2026-09-25").

Serialization uses skops, the same convention as the existing MLflow
logger (src/tracking/mlflow_logger.py): no pickle, and any untrusted
type surfaced by skops on load must be checked explicitly rather than
blanket-trusted.

required_raw_history lives here (not in src/serving/buffer.py, which
needs it too) because the bundle schema must be able to record it at
training time, before the serving buffer module exists at runtime;
buffer.py imports it from here rather than redefining it.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import skops.io as sio
import torch
from sklearn.preprocessing import StandardScaler

from config.features import FEATURE_COLUMNS, LAG_STEPS, ROLLING_WINDOWS, SCALED_COLUMNS
from src.models.forecaster import Forecaster
from src.models.lstm import LSTMForecaster
from src.models.sklearn_models import LinearRegressionForecaster, RandomForestForecaster

# Sequence-model families serialize via torch state_dict + architecture
# reconstruction (below), never via skops — deliberately a separate path
# from the sklearn estimators' skops serialization. Only "lstm" so far;
# gru/cnn_lstm are deferred to a follow-up (DECISIONS.md "Phase 6:
# deep-model final deployment decision, 2026-09-25").
_SEQUENCE_FAMILIES = {"lstm"}


def required_raw_history(
    lag_steps: list[int] | None = None,
    rolling_windows: list[int] | None = None,
) -> int:
    """Minimum contiguous raw observations needed to produce one feature row."""
    if lag_steps is None:
        lag_steps = LAG_STEPS
    if rolling_windows is None:
        rolling_windows = ROLLING_WINDOWS
    return max(max(lag_steps), max(rolling_windows)) + 1


def feature_schema_version(feature_columns: list[str]) -> str:
    """First 12 hex chars of the sha256 of the ordered feature columns, comma-joined."""
    digest = hashlib.sha256(",".join(feature_columns).encode("utf-8")).hexdigest()
    return digest[:12]


@dataclass(frozen=True)
class ModelBundle:
    forecaster: Forecaster
    scaler: StandardScaler
    schema: dict


def _dump_skops(obj, path: Path) -> None:
    sio.dump(obj, path)


# Scoped, explicit trust per model family — matching the convention
# already established in src/tracking/mlflow_logger.py: skops flags
# sklearn.tree._tree.Tree by default for tree-based models (a malicious
# file could set out-of-bounds node indices), which is not a concern for
# models this project trains and saves itself. Trust is scoped to exactly
# the flagged type per family, never a blanket override.
_SKOPS_TRUSTED_TYPES: dict[str, list[str]] = {
    "linear_regression": [],
    "random_forest": ["sklearn.tree._tree.Tree"],
}


def _load_skops_trusted(path: Path, trusted: list[str] | None = None):
    trusted = trusted or []
    untrusted = sio.get_untrusted_types(file=path)
    unexpected = [t for t in untrusted if t not in trusted]
    if unexpected:
        raise ValueError(
            f"skops file {path} contains untrusted types not explicitly reviewed: {unexpected}"
        )
    return sio.load(path, trusted=trusted)


def _save_torch_state(forecaster, path: Path) -> None:
    """Saves only the trained network's state_dict — never the whole Forecaster
    object, never pickle. Mirrors the "never pickle" discipline of the skops
    path used for the sklearn-based families."""
    torch.save(forecaster._net.state_dict(), path)


def save_bundle(bundle: ModelBundle, directory: Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if bundle.schema["model_family"] in _SEQUENCE_FAMILIES:
        _save_torch_state(bundle.forecaster, directory / "model_state.pt")
    else:
        _dump_skops(bundle.forecaster.model, directory / "estimator.skops")

    _dump_skops(bundle.scaler, directory / "scaler.skops")
    with open(directory / "schema.json", "w") as f:
        json.dump(bundle.schema, f, indent=2, sort_keys=True)


def _load_linear_regression(directory: Path, schema: dict) -> Forecaster:
    estimator = _load_skops_trusted(
        directory / "estimator.skops", trusted=_SKOPS_TRUSTED_TYPES["linear_regression"]
    )
    forecaster = LinearRegressionForecaster(fit_intercept=bool(estimator.fit_intercept))
    forecaster.model = estimator
    return forecaster


def _load_random_forest(directory: Path, schema: dict) -> Forecaster:
    estimator = _load_skops_trusted(
        directory / "estimator.skops", trusted=_SKOPS_TRUSTED_TYPES["random_forest"]
    )
    forecaster = RandomForestForecaster(
        n_estimators=estimator.n_estimators,
        max_depth=estimator.max_depth,
        random_state=estimator.random_state,
    )
    forecaster.model = estimator
    return forecaster


def _load_lstm(directory: Path, schema: dict) -> Forecaster:
    """Never touches skops: reconstructs the forecaster from its recorded
    constructor params (schema["architecture"]["params"], the same values
    that produced the trained network — sourced from forecaster.params,
    the existing source of truth), builds an empty network via the
    forecaster's own _build_network, then loads the saved state_dict into
    it. This is the same way a fitted SequenceForecaster normally holds
    its trained network (self._net), not a guess."""
    architecture = schema["architecture"]
    forecaster = LSTMForecaster(**architecture["params"])
    net = forecaster._build_network(architecture["n_features"])
    state_dict = torch.load(directory / "model_state.pt", weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    forecaster._net = net
    return forecaster


LOADERS: dict[str, Callable[[Path, dict], Forecaster]] = {
    "linear_regression": _load_linear_regression,
    "random_forest": _load_random_forest,
    "lstm": _load_lstm,
}


def load_bundle(directory: Path) -> ModelBundle:
    directory = Path(directory)
    with open(directory / "schema.json") as f:
        schema = json.load(f)

    if schema["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError(
            "bundle schema feature_columns differs from config/features.py FEATURE_COLUMNS"
        )
    if schema["scaled_columns"] != SCALED_COLUMNS:
        raise ValueError(
            "bundle schema scaled_columns differs from config/features.py SCALED_COLUMNS"
        )

    model_family = schema["model_family"]
    if model_family not in LOADERS:
        raise ValueError(f"no loader registered for model_family={model_family!r}")

    forecaster = LOADERS[model_family](directory, schema)
    scaler = _load_skops_trusted(directory / "scaler.skops")

    return ModelBundle(forecaster=forecaster, scaler=scaler, schema=schema)
