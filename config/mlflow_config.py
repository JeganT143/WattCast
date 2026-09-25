"""
MLflow tracking configuration — single source of truth.

Every module or notebook that talks to MLflow MUST import TRACKING_URI
from here and call mlflow.set_tracking_uri(TRACKING_URI) explicitly.
Never rely on MLflow's default: it silently creates a disconnected store
relative to whichever directory the process happens to run from (this
caused real store fragmentation during development — see decisions.md,
ADR-010).

The path is absolute, built from PROJECT_ROOT, so the store location is
identical whether the importing code runs from a notebook, a script at
the repo root, or a pytest process.

MLflow is a training-time dependency only: the Streamlit app and the
FastAPI service load the exported bundles in models/ and never import it.
"""

from config.paths import PROJECT_ROOT

TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'db' / 'mlflow.db'}"
EXPERIMENT_NAME = "WattCast"
