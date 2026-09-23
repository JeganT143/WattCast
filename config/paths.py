"""
Filesystem path configuration — single source of truth for locating
project data on disk, independent of the caller's working directory.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw Data
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "energy_data_set.csv"