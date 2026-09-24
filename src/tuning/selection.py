"""The pure validation-only selection rule from the Phase 4 protocol. score_configurations and
select_configuration do no I/O. load_val_reference reads the golden fixture and returns ONLY the
two linear_regression|h6 validation metrics; it indexes val_mae and val_rmse and never returns or
uses any test value (the file holds them, so the parse loads them, but nothing here touches them).
"""

import math
from pathlib import Path
import json

_REFERENCE_KEY = "linear_regression|h6"
_REQUIRED = ("max_epochs", "huber_delta", "mean_val_mae", "mean_val_rmse")


def _check_reference(name, value):
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"reference {name} must be a positive finite number, got {value!r}")


def score_configurations(rows, lr_val_mae, lr_val_rmse):
    _check_reference("lr_val_mae", lr_val_mae)
    _check_reference("lr_val_rmse", lr_val_rmse)

    for row in rows:
        missing = [k for k in _REQUIRED if k not in row]
        if missing:
            raise ValueError(f"row is missing keys {missing}")

    for row in rows:
        if not math.isfinite(row["mean_val_mae"]) or not math.isfinite(row["mean_val_rmse"]):
            raise ValueError(
                f"non-finite validation metric in configuration {(row['max_epochs'], row['huber_delta'])}"
            )

    seen = set()
    for row in rows:
        key = (row["max_epochs"], row["huber_delta"])
        if key in seen:
            raise ValueError(f"duplicate configuration {key}")
        seen.add(key)

    result = []
    for row in rows:
        val_mae_ratio = row["mean_val_mae"] / lr_val_mae
        val_rmse_ratio = row["mean_val_rmse"] / lr_val_rmse
        result.append({
            **row,
            "val_mae_ratio": val_mae_ratio,
            "val_rmse_ratio": val_rmse_ratio,
            "selection_score": max(val_mae_ratio, val_rmse_ratio),
        })
    return result


def select_configuration(rows, lr_val_mae, lr_val_rmse):
    scored = score_configurations(rows, lr_val_mae, lr_val_rmse)
    if not scored:
        raise ValueError("no configurations to select from (empty input)")
    return min(
        scored,
        key=lambda r: (
            r["selection_score"],
            r["val_mae_ratio"],
            r["val_rmse_ratio"],
            r["max_epochs"],
            r["huber_delta"],
        ),
    )


def load_val_reference(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entry = data.get("results", {}).get(_REFERENCE_KEY)
    if entry is None:
        raise ValueError(f"fixture has no {_REFERENCE_KEY!r} entry")
    metrics = entry["metrics"]
    return {"lr_val_mae": float(metrics["val_mae"]), "lr_val_rmse": float(metrics["val_rmse"])}
