# WattCast

Household appliance energy-consumption forecasting on the [UCI Appliances Energy Prediction dataset](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) (Candanedo et al.) — 19,735 rows, 29 columns, 10-minute sensor intervals spanning ~4.5 months (2016-01-11 to 2016-05-27). The project builds a leakage-safe feature pipeline and a `Forecaster` strategy interface to compare naive and learned baselines at multiple forecast horizons, tracked end-to-end with DVC + MLflow.

## Repo structure

```
config/                   Single source of truth for paths, feature schema, MLflow config
  paths.py                  PROJECT_ROOT, RAW_DATA_PATH
  features.py                Lag/rolling/window sizes, split boundaries, MAPE threshold
  mlflow_config.py           MLflow tracking URI, artifact root, experiment name
src/
  data/
    assemble.py               Horizon-specific modeling dataset assembly
    split.py                   Chronological train/val/test boolean masks
  features/
    lag.py, rolling.py         Backward-only lag/rolling feature construction
    temporal.py                 Cyclical calendar feature encoding
    targets.py                   Forward-shifted target construction
    build_features.py           Canonical feature-engineering orchestration
  preprocessing/
    scaling.py                 Train-only-fit StandardScaler fit/transform
  pipeline.py                Full Phase 2 pipeline: raw data -> modeling datasets
  models/
    forecaster.py               Forecaster ABC (Strategy pattern)
    naive.py                    Persistence and seasonal-naive baselines
    sklearn_models.py           Linear Regression / Random Forest adapters
  evaluation/
    harness.py                  evaluate_forecaster(): fit once, score train/val/test
    metrics.py                   mae, rmse, mape
  tracking/
    mlflow_logger.py            Logs an EvaluationResult to MLflow
notebooks/                 EDA, feature-engineering exploration, baseline leaderboard
tests/                     pytest suite (106 tests, mirrors src/ structure)
data/                       DVC-tracked raw + processed data (not in Git)
db/mlflow.db                 MLflow SQLite tracking store (gitignored)
DECISIONS.md                Full methodology, findings, and decisions log
```

## Setup

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt

dvc pull          # fetches data/raw and data/processed from the configured DVC remote
```

MLflow tracks to a local SQLite store at `db/mlflow.db` (see `config/mlflow_config.py`); no separate server needs to be running for tests or pipeline runs.

## Running the tests

```bash
python -m pytest -v
```

## Reproducing the Phase 2 feature pipeline

```python
import pandas as pd
from config.paths import RAW_DATA_PATH
from src.pipeline import run_pipeline

df_raw = pd.read_csv(RAW_DATA_PATH, parse_dates=["date"])
result = run_pipeline(df_raw)   # PipelineResult: scaler + train/val/test x {t1, t6}
```

Equivalently, `notebooks/02_feature_engineering_exploration.ipynb` walks through the same pipeline with inline leakage checks.

## Reproducing the Phase 3 baseline leaderboard

Run `notebooks/03_baselines.ipynb`, which for each forecast horizon in `TARGET_HORIZONS`:
1. runs `src.pipeline.run_pipeline` on the raw CSV to get the Phase 2 train/val/test datasets,
2. fits `NaivePersistenceForecaster`, `NaiveSeasonalForecaster`, `LinearRegressionForecaster`, and `RandomForestForecaster` via `src.evaluation.harness.evaluate_forecaster`,
3. logs each run to MLflow (`src.tracking.mlflow_logger.log_evaluation_result`) under the `WattCast` experiment.

View results with:

```bash
mlflow ui --backend-store-uri sqlite:///db/mlflow.db
```

## Project status

- **Phase 0** — MLOps foundation (DVC + MLflow) — complete
- **Phase 1** — Deep EDA — complete
- **Phase 2** — Leakage-checked feature engineering and preprocessing pipeline — complete
- **Phase 3** — Forecaster interface, naive + sklearn baselines, evaluation harness, MLflow integration — complete
- **Phase 4/5** — PyTorch sequence models, walk-forward validation — not yet started

See [`DECISIONS.md`](DECISIONS.md) for full methodology, findings, and the decisions log across all phases.
