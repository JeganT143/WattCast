"""Tests for scripts/train_final_lr.py's guard logic and importability
(DECISIONS.md "Phase 6: serving registration, 2026-09-25")."""

import pytest

from scripts import train_final_lr


def test_main_is_importable_and_callable():
    assert callable(train_final_lr.main)
    assert callable(train_final_lr.check_can_run)


def test_refuses_when_results_path_exists(tmp_path):
    results_path = tmp_path / "final_lr_registration.json"
    results_path.write_text("{}")

    with pytest.raises(FileExistsError):
        train_final_lr.check_can_run(results_path, tracking_uri="sqlite:///unused.db")
