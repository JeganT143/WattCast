"""
Filesystem path configuration — single source of truth for locating
project data on disk, independent of the caller's working directory.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw data (UCI Appliances Energy Prediction; fetched by scripts/download_data.py)
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "energy_data_set.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Committed serving artifacts: one bundle directory per model family, plus the
# pre-test history the rolling buffer is seeded with at startup
# (exported from the MLflow registry by scripts/export_serving_models.py).
MODELS_DIR = PROJECT_ROOT / "models"
SEED_HISTORY_PATH = MODELS_DIR / "seed_history.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
