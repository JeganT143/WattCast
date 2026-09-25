"""Skips tests whose local-only inputs are absent, so a fresh clone runs the
suite green. Mark a test with the resource it needs:

- requires_raw_data        data/raw/energy_data_set.csv   (make data)
- requires_processed_data  data/processed/*.csv           (make data)
- requires_registry        MLflow registry in db/mlflow.db (training runs)
"""

import pytest

from config.paths import PROCESSED_DIR, PROJECT_ROOT, RAW_DATA_PATH

_RESOURCES = {
    "requires_raw_data": (RAW_DATA_PATH, "raw dataset not present (run `make data`)"),
    "requires_processed_data": (PROCESSED_DIR / "train_t6.csv", "processed data not present (run `make data`)"),
    "requires_registry": (PROJECT_ROOT / "db" / "mlflow.db", "local MLflow registry not present"),
}


def pytest_configure(config):
    for marker, (_, reason) in _RESOURCES.items():
        config.addinivalue_line("markers", f"{marker}: skip unless present — {reason}")


def pytest_runtest_setup(item):
    for marker, (path, reason) in _RESOURCES.items():
        if item.get_closest_marker(marker) and not path.exists():
            pytest.skip(reason)
