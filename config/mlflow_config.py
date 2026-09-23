"""
MLflow tracking configuration — single source of truth.

Every module or notebook that talks to MLflow MUST import TRACKING_URI
from here and call mlflow.set_tracking_uri(TRACKING_URI) explicitly,
every time, at the top of the session. Never rely on MLflow's default —
it silently creates a disconnected store relative to whichever directory
the process happens to be running from (confirmed: this caused real
fragmentation across db/mlflow.db, repo-root mlflow.db, and
notebooks/mlflow.db during Phase 3 development — see decisions log).

Absolute paths only, built from PROJECT_ROOT, so the store location is
identical regardless of whether the importing code runs from a notebook
in notebooks/, a script at the repo root, or a pytest process.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'db' / 'mlflow.db'}"
ARTIFACT_ROOT = str(PROJECT_ROOT / "models")
EXPERIMENT_NAME = "WattCast"
