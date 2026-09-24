"""The pre-registered final evaluation. It takes no seeds or horizons parameter: the registered
seeds and horizons are constants. It reads the selected configuration from a committed tuning
result, fits one fresh LSTMForecaster per (registered horizon, registered seed) and evaluates
each once through the full three-partition evaluation, computes the Tier 1 verdict with the
registered comparison, reports every seed at both horizons whatever the outcome, and refuses to
overwrite an earlier output. It makes no choice among seeds, horizons, or models.
"""

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd

from config.features import MAPE_THRESHOLD
from src.evaluation.harness import evaluate_forecaster
from src.models.lstm import LSTMForecaster

REGISTERED_SEEDS = (42, 43, 44)
REGISTERED_HORIZONS = (6, 1)
TIER1_FACTOR = 0.99


def tier1_verdict(seed_results, ref_mae, ref_rmse) -> dict:
    seeds = sorted(int(r["seed"]) for r in seed_results)
    if seeds != list(REGISTERED_SEEDS):
        raise ValueError(f"tier1_verdict needs exactly the registered seeds {list(REGISTERED_SEEDS)}, got {seeds}")

    threshold_mae = TIER1_FACTOR * ref_mae
    threshold_rmse = TIER1_FACTOR * ref_rmse
    per_seed = []
    for r in sorted(seed_results, key=lambda r: r["seed"]):
        mae_pass = bool(r["test_mae"] <= threshold_mae)
        rmse_pass = bool(r["test_rmse"] <= threshold_rmse)
        per_seed.append(
            {"seed": int(r["seed"]), "mae_pass": mae_pass, "rmse_pass": rmse_pass, "pass": mae_pass and rmse_pass}
        )
    return {
        "factor": TIER1_FACTOR,
        "threshold_mae": threshold_mae,
        "threshold_rmse": threshold_rmse,
        "per_seed": per_seed,
        "shown": all(p["pass"] for p in per_seed),
    }


def load_tier1_reference(fixture_path) -> dict:
    data = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    entry = data.get("results", {}).get("linear_regression|h6")
    if entry is None:
        raise ValueError("fixture has no 'linear_regression|h6' entry")
    ref_mae = float(entry["metrics"]["test_mae"])
    ref_rmse = float(entry["metrics"]["test_rmse"])
    return {
        "ref_mae": ref_mae,
        "ref_rmse": ref_rmse,
        "threshold_mae": TIER1_FACTOR * ref_mae,
        "threshold_rmse": TIER1_FACTOR * ref_rmse,
    }


def load_selected_configuration(results_path) -> dict:
    data = json.loads(Path(results_path).read_text(encoding="utf-8"))
    sel = data["selected"]
    return {
        "max_epochs": int(sel["max_epochs"]),
        "huber_delta": float(sel["huber_delta"]),
        "model_kwargs": dict(data["protocol"]["model_params"]),
    }


def summarize(values) -> dict:
    values = [float(v) for v in values]
    if len(values) < 2:
        raise ValueError("summarize needs at least two values")
    return {"mean": statistics.fmean(values), "std": statistics.stdev(values)}


def run_final_evaluation(*, data_dir, tuning_results_path, reference_path, out_path, predictions_path) -> dict:
    out_path = Path(out_path)
    predictions_path = Path(predictions_path)
    for p in (out_path, predictions_path):
        if p.exists():
            raise FileExistsError(f"{p} already exists; the final evaluation runs once")

    cfg = load_selected_configuration(tuning_results_path)
    ref = load_tier1_reference(reference_path)
    fixture = json.loads(Path(reference_path).read_text(encoding="utf-8"))["results"]

    horizons = {}
    predictions = None
    for h in REGISTERED_HORIZONS:
        target = f"target_t{h}"
        train_df = pd.read_csv(Path(data_dir) / f"train_t{h}.csv", parse_dates=["date"])
        val_df = pd.read_csv(Path(data_dir) / f"val_t{h}.csv", parse_dates=["date"])
        test_df = pd.read_csv(Path(data_dir) / f"test_t{h}.csv", parse_dates=["date"])

        runs = []
        preds_by_seed = {}
        actual = None
        for seed in REGISTERED_SEEDS:
            forecaster = LSTMForecaster(
                max_epochs=cfg["max_epochs"], huber_delta=cfg["huber_delta"], seed=seed, **cfg["model_kwargs"]
            )
            result = evaluate_forecaster(
                forecaster,
                train_df,
                val_df,
                test_df,
                target_column=target,
                model_name="lstm",
                horizon=h,
                mape_threshold=MAPE_THRESHOLD,
            )
            runs.append(
                {
                    "seed": int(seed),
                    "params": forecaster.params,
                    "test_mae": float(result.metrics["test"]["mae"]),
                    "test_rmse": float(result.metrics["test"]["rmse"]),
                    "test_mape": float(result.metrics["test"]["mape"]),
                    "n_evaluated": {k: int(v) for k, v in result.n_evaluated.items()},
                    "n_warmup_rows": int(result.n_warmup_rows),
                    "n_ineligible_label_rows": int(result.n_ineligible_label_rows),
                }
            )
            preds_by_seed[seed] = result.predictions["test"]
            actual = result.actuals["test"]

        horizons[str(h)] = {
            "runs": runs,
            "summary": {m: summarize([r[m] for r in runs]) for m in ("test_mae", "test_rmse", "test_mape")},
        }
        if h == 6:
            if any(len(p) != len(test_df) for p in preds_by_seed.values()):
                raise ValueError("every test row must be scored for the predictions file")
            predictions = pd.DataFrame(
                {
                    "date": test_df["date"].reset_index(drop=True),
                    "actual": actual,
                    **{f"pred_seed_{s}": preds_by_seed[s] for s in REGISTERED_SEEDS},
                }
            )

    runs6 = horizons["6"]["runs"]
    tier1 = tier1_verdict(
        [{"seed": r["seed"], "test_mae": r["test_mae"], "test_rmse": r["test_rmse"]} for r in runs6],
        ref["ref_mae"],
        ref["ref_rmse"],
    )

    baselines = {
        key: {m: float(entry["metrics"][m]) for m in ("test_mae", "test_rmse", "test_mape")}
        for key, entry in fixture.items()
    }

    result = {
        "protocol": {
            "seeds": list(REGISTERED_SEEDS),
            "horizons": list(REGISTERED_HORIZONS),
            "tier1_factor": TIER1_FACTOR,
            "selected": {"max_epochs": cfg["max_epochs"], "huber_delta": cfg["huber_delta"]},
            "model_params": cfg["model_kwargs"],
            "mape_threshold": float(MAPE_THRESHOLD),
            "data_dir": str(data_dir),
            "tuning_results_path": str(tuning_results_path),
            "reference_path": str(reference_path),
        },
        "tier1_reference": ref,
        "tier1": tier1,
        "horizons": horizons,
        "baselines": baselines,
    }

    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)
    out_path.write_text(text, encoding="utf-8")
    return json.loads(text)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the pre-registered final LSTM evaluation.")
    parser.add_argument("--data-dir", required=True, help="directory holding the six horizon partition files")
    parser.add_argument("--tuning-results", required=True, help="path to the committed tuning result")
    parser.add_argument("--reference", required=True, help="path to the golden baseline fixture")
    parser.add_argument("--out", required=True, help="path to write the final evaluation result to")
    parser.add_argument("--predictions", required=True, help="path to write the horizon-6 predictions to")
    args = parser.parse_args(argv)

    result = run_final_evaluation(
        data_dir=args.data_dir,
        tuning_results_path=args.tuning_results,
        reference_path=args.reference,
        out_path=args.out,
        predictions_path=args.predictions,
    )
    v = result["tier1"]
    print(
        "Tier 1 (h=6, all three seeds, both metrics <= 0.99 x the linear_regression h=6 reference):",
        "SHOWN" if v["shown"] else "NOT SHOWN",
    )
    for p, r in zip(v["per_seed"], result["horizons"]["6"]["runs"]):
        print(
            f"  seed {p['seed']}: test_mae={r['test_mae']:.3f} (threshold {v['threshold_mae']:.3f}, pass={p['mae_pass']}) "
            f"test_rmse={r['test_rmse']:.3f} (threshold {v['threshold_rmse']:.3f}, pass={p['rmse_pass']})"
        )
    print("wrote", args.out)
    print("wrote", args.predictions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
