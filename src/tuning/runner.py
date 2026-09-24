"""The validation-only grid runner for the pre-registered protocol. It receives paths to a
training file, a validation file and a reference fixture, fits one fresh LSTMForecaster per
(max_epochs, huber_delta, seed) through the validation-only evaluator, averages the validation
metrics over seeds per configuration, applies the registered selection rule, and writes one JSON
with validation values only. No other partition path exists in this module and nothing is logged
to any tracking service.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluation.validation_only import evaluate_on_validation
from src.models.lstm import LSTMForecaster
from src.tuning.selection import load_val_reference, score_configurations, select_configuration

_GRID_CONTROLLED = ("max_epochs", "huber_delta", "seed")


def run_grid(
    train_path,
    val_path,
    reference_path,
    out_path,
    *,
    target_column="target_t6",
    seeds=(42, 43, 44),
    epochs=(10, 20, 35, 50),
    deltas=(20.0, 40.0, 60.0),
    model_kwargs=None,
) -> dict:
    model_kwargs = dict(model_kwargs or {})
    clash = sorted(set(model_kwargs) & set(_GRID_CONTROLLED))
    if clash:
        raise ValueError(f"model_kwargs must not set grid-controlled parameters: {clash}")

    train_df = pd.read_csv(train_path, parse_dates=["date"])
    val_df = pd.read_csv(val_path, parse_dates=["date"])
    ref = load_val_reference(reference_path)

    fixed = {
        k: v
        for k, v in LSTMForecaster(
            max_epochs=epochs[0], huber_delta=deltas[0], seed=seeds[0], **model_kwargs
        ).params.items()
        if k not in _GRID_CONTROLLED
    }

    runs = []
    total = len(epochs) * len(deltas) * len(seeds)
    for max_epochs in epochs:
        for huber_delta in deltas:
            for seed in seeds:
                forecaster = LSTMForecaster(
                    max_epochs=max_epochs, huber_delta=huber_delta, seed=seed, **model_kwargs
                )
                result = evaluate_on_validation(forecaster, train_df, val_df, target_column)
                runs.append(
                    {
                        "max_epochs": int(max_epochs),
                        "huber_delta": float(huber_delta),
                        "seed": int(seed),
                        "val_mae": float(result.metrics["mae"]),
                        "val_rmse": float(result.metrics["rmse"]),
                        "n_evaluated": int(result.n_evaluated),
                        "n_warmup_rows": int(result.n_warmup_rows),
                        "n_ineligible_label_rows": int(result.n_ineligible_label_rows),
                    }
                )
                print(
                    f"[{len(runs)}/{total}] max_epochs={max_epochs} huber_delta={huber_delta} seed={seed} "
                    f"val_mae={runs[-1]['val_mae']:.3f} val_rmse={runs[-1]['val_rmse']:.3f}",
                    flush=True,
                )

    rows = []
    for max_epochs in epochs:
        for huber_delta in deltas:
            group = [
                r for r in runs if r["max_epochs"] == int(max_epochs) and r["huber_delta"] == float(huber_delta)
            ]
            rows.append(
                {
                    "max_epochs": int(max_epochs),
                    "huber_delta": float(huber_delta),
                    "mean_val_mae": sum(r["val_mae"] for r in group) / len(group),
                    "mean_val_rmse": sum(r["val_rmse"] for r in group) / len(group),
                }
            )

    configurations = score_configurations(rows, ref["lr_val_mae"], ref["lr_val_rmse"])
    selected = select_configuration(rows, ref["lr_val_mae"], ref["lr_val_rmse"])

    result = {
        "protocol": {
            "target_column": target_column,
            "seeds": [int(s) for s in seeds],
            "epochs": [int(e) for e in epochs],
            "deltas": [float(d) for d in deltas],
            "model_params": fixed,
            "train_path": str(train_path),
            "val_path": str(val_path),
            "reference_path": str(reference_path),
        },
        "reference": ref,
        "runs": runs,
        "configurations": configurations,
        "selected": selected,
    }

    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(text, encoding="utf-8")
    return json.loads(text)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fit the validation-only grid for the LSTM protocol.")
    parser.add_argument("--train", required=True, help="path to the training file")
    parser.add_argument("--val", required=True, help="path to the validation file")
    parser.add_argument("--reference", required=True, help="path to the reference fixture")
    parser.add_argument("--out", required=True, help="path to write the result to")
    parser.add_argument("--target", default="target_t6", help="name of the label column")
    parser.add_argument(
        "--epochs", nargs="+", type=int, default=[10, 20, 35, 50], help="epoch counts to sweep"
    )
    parser.add_argument(
        "--deltas", nargs="+", type=float, default=[20.0, 40.0, 60.0], help="Huber delta values to sweep"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 43, 44], help="random seeds to average over"
    )
    args = parser.parse_args(argv)

    result = run_grid(
        args.train,
        args.val,
        args.reference,
        args.out,
        target_column=args.target,
        seeds=args.seeds,
        epochs=args.epochs,
        deltas=args.deltas,
    )
    sel = result["selected"]
    print(
        "selected:",
        json.dumps(
            {k: sel[k] for k in ("max_epochs", "huber_delta", "selection_score", "val_mae_ratio", "val_rmse_ratio")},
            sort_keys=True,
        ),
    )
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
