# Engineering decisions

This file records the decisions that shape WattCast: why each was made, the alternatives considered, and what each costs. It is a curated summary. The dated, append-only log written during development, including the pre-registrations committed before their experiments ran, is kept verbatim in [`docs/research-log.md`](docs/research-log.md).

Each record uses the same fields: **Context**, **Options considered**, **Decision**, **Reasoning**, **Trade-offs**, **Consequences**. Code comments cite records by number (`decisions.md, ADR-014`).

## Project phases

The code and the research log refer to these phases.

| Phase | Scope | When |
|---|---|---|
| 0 | DVC + MLflow foundation | 2026-09-23 |
| 1 | Exploratory data analysis | 2026-09-23 |
| 2 | Leakage-safe features and preprocessing | 2026-09-23 |
| 3 | `Forecaster` interface, baselines, evaluation harness, MLflow logging | 2026-09-23/24 |
| 4 | LSTM with a pre-registered protocol and one held-out test check | 2026-09-24 |
| 5 | GRU and CNN-LSTM; pre-registered 8-fold walk-forward comparison | 2026-09-24 |
| 6 | Final training, model registry, FastAPI serving, Streamlit app | 2026-09-25 |
| — | Public-release packaging (ADR-016 to ADR-018) | 2026-09-25 |

## Index

| # | Decision | Area |
|---|---|---|
| [001](#adr-001--true-forecasting-with-leakage-rules-enforced-in-code) | True forecasting; leakage rules enforced in code | Data & features |
| [002](#adr-002--features-before-splitting-chronological-split-train-only-scaling) | Features before splitting; chronological split; train-only scaling | Data & features |
| [003](#adr-003--keep-the-raw-target) | Keep the raw target | Data & features |
| [004](#adr-004--one-forecaster-interface-for-every-model) | One `Forecaster` interface for every model | Modelling |
| [005](#adr-005--shared-sequence-model-base-with-one-fixed-configuration) | Shared sequence-model base with one fixed configuration | Modelling |
| [006](#adr-006--mae-and-rmse-decide-mape-is-reported-only) | MAE and RMSE decide; MAPE is reported only | Evaluation |
| [007](#adr-007--pre-register-evaluation-rules-use-the-test-partition-once) | Pre-register evaluation rules; use the test partition once | Evaluation |
| [008](#adr-008--a-structurally-validation-only-tuning-path) | A structurally validation-only tuning path | Evaluation |
| [009](#adr-009--walk-forward-comparison-design) | Walk-forward comparison design | Evaluation |
| [010](#adr-010--one-mlflow-store-aliases-not-stages-tracking-outside-the-core) | One MLflow store, aliases not stages, tracking outside the core | Tracking |
| [011](#adr-011--serialize-models-without-pickle) | Serialize models without pickle | Artifacts |
| [012](#adr-012--linear-regression-as-primary-serving-model-all-five-families-served) | Linear regression as primary serving model; all five families served | Serving |
| [013](#adr-013--final-models-train-on-train--validation-never-on-test) | Final models train on train + validation, never on test | Serving |
| [014](#adr-014--stateful-serving-with-a-seeded-rolling-buffer) | Stateful serving with a seeded rolling buffer | Serving |
| [015](#adr-015--pre-registered-serving-equivalence-bars) | Pre-registered serving-equivalence bars | Serving |
| [016](#adr-016--ship-exported-model-bundles-in-the-repository) | Ship exported model bundles in the repository | Deployment |
| [017](#adr-017--streamlit-runs-inference-in-process-with-per-session-state) | Streamlit runs inference in-process with per-session state | Deployment |
| [018](#adr-018--reproducible-data-from-the-public-source) | Reproducible data from the public source | Deployment |

---

## Data & features

### ADR-001 · True forecasting with leakage rules enforced in code

*Phase 1–2 · Accepted*

- **Context.** The target is `Appliances` energy (Wh) per 10-minute interval. The dataset also has same-timestamp sensor readings (room temperature and humidity, weather, lights). While building features we noticed that `lights` measured at the target time would leak information, and the same holds for every contemporaneous sensor.
- **Options considered.** (1) Nowcasting: allow same-timestamp sensors. (2) True forecasting: use only information available at the forecast origin *t* to predict *t + h*.
- **Decision.** True forecasting at horizons h = 1 and h = 6 (10 and 60 minutes). h = 6 became the primary horizon in Phase 4. Features are past `Appliances` values (lags 1–6 and 144; rolling mean/std over 6 and 18 steps) plus calendar features. Calendar values are known in advance, so they carry no leakage risk.
- **Reasoning.** A live service only knows the past. The rule is enforced structurally: lag and rolling helpers reject non-positive offsets, rolling windows are never centred, and target construction rejects non-positive horizons (`src/features/`).
- **Trade-offs.** Lagged sensor values could be legitimate features but were not added. Same-timestamp correlations with the target were all weak (|r| < 0.25).
- **Consequences.** Every model sees the same 18-column feature contract (`config/features.py`), and serving can build features from a buffer of `Appliances` readings alone.

### ADR-002 · Features before splitting; chronological split; train-only scaling

*Phase 2 · Accepted*

- **Context.** Time series need chronological evaluation. Lag features at a partition's start need history from the previous partition.
- **Options considered.** (1) Random split: leaks the future. (2) Split first, then build features: the first 144 rows of each partition lose their daily lag. (3) Build features on the full continuous series, then split with boolean masks.
- **Decision.** Option 3. Partitions: train < 2016-04-18 ≤ validation < 2016-04-30 ≤ test. A `StandardScaler` is fit on training rows only (11 lag/rolling columns) and applied unchanged everywhere else. Training labels whose target time falls in the next partition are masked to NaN; the rows themselves are kept (`src/data/purge.py`).
- **Reasoning.** Lags see real history across boundaries. Validation and test statistics never reach the scaler. Keeping purged rows avoids opening gaps in sequence windows.
- **Trade-offs.** Masking instead of dropping needs extra bookkeeping in the harness: a trailing block of non-finite labels is excluded from fitting.
- **Consequences.** `transform_with_scaler` knows nothing about partitions, so serving reuses it unchanged. Walk-forward folds refit their own train-only scaler (ADR-009).

### ADR-003 · Keep the raw target

*Phase 1 · Accepted*

- **Context.** The target is right-skewed (median 60 Wh, maximum 1,080 Wh). A rolling-IQR rule flagged 11.26% of readings as outliers. STL showed a daily rhythm with stable shape but declining amplitude.
- **Options considered.** Clip or remove extremes; difference the series; keep it unmodified.
- **Decision.** Keep the target unmodified: no clipping, no removal, no differencing.
- **Reasoning.** 11% is too many to be rare anomalies, and every inspected extreme sits inside a plausible ramp. The ADF test rejects a unit root (statistic −21.62), and none of the models requires stationarity.
- **Trade-offs.** Large spikes stay in the data and dominate squared-error metrics (see ADR-006).
- **Consequences.** High-consumption events are signal the models must learn. They are also where deep and classical models differ most (the MAE/RMSE split).

---

## Modelling

### ADR-004 · One `Forecaster` interface for every model

*Phase 3 · Accepted*

- **Context.** Naive baselines, sklearn models and PyTorch sequence models had to be trained, evaluated and served by the same code.
- **Options considered.** (1) Model-specific branches in the harness and serving code. (2) A shared global feature list. (3) A small abstract interface where each model declares its own inputs.
- **Decision.** `Forecaster` (Strategy pattern) with `fit(X, y)`, `predict(X)` returning a 1-D array, `params`, `required_columns` and `required_history_length` (0 for row-wise models; L − 1 for sequence models, whose first L − 1 outputs are NaN).
- **Reasoning.** The persistence baseline needs `Appliances[t]`, which is deliberately not a learned-model feature, so inputs must be declared per model. With the interface, the evaluation harness, the walk-forward runner and the serving core never branch on model type.
- **Trade-offs.** Callers build `X` from `required_columns` and supply the preceding context rows themselves.
- **Consequences.** Adding a model family means writing one subclass and one bundle loader. MLflow logging stays outside the interface (ADR-010).

### ADR-005 · Shared sequence-model base with one fixed configuration

*Phase 4–5 · Accepted*

- **Context.** LSTM, GRU and CNN-LSTM share windowing, seeding, threading and the training loop. Measurements showed that the torch thread count changes results: 1 thread vs 4 threads differed by up to 35.36 Wh after 3 epochs.
- **Options considered.** Independent implementations per model, or one base class with a per-model network builder. Tune each model separately, or reuse one pre-registered configuration.
- **Decision.** `SequenceForecaster` implements everything except `_build_network`. `make_windows` is the single causal windowing function for both training and inference. One configuration: window L = 18, hidden 64, 1 layer, no dropout, Adam (lr 1e-3), batch 64, Huber loss (δ = 40 Wh), 50 epochs, no early stopping, output bias initialised to the training-target median, 4 torch threads, 16 input columns (the raw hour and weekday integers are dropped). Only the LSTM was tuned: a 12-configuration grid over epochs and Huber δ, 3 seeds each, selected on validation only.
- **Reasoning.** One windowing function removes train/serve skew. Fixing the thread count makes runs repeatable (two 4-thread fits were bit-identical). A frozen LSTM golden fixture proved the refactor into the base class changed nothing.
- **Trade-offs.** GRU and CNN-LSTM run untuned, so the tuning advantage is one-sided (a registered limitation).
- **Consequences.** All three architectures serialize, reload and serve through one code path (ADR-011).

---

## Evaluation

### ADR-006 · MAE and RMSE decide; MAPE is reported only

*Phase 3–5 · Accepted*

- **Context.** Consumption never reaches 0 Wh, but low readings (floor 10 Wh) make percentage errors unstable.
- **Options considered.** A single metric, or several metrics with defined roles.
- **Decision.** MAE and RMSE are both decision-making, computed over all rows. MAPE is reported only, and it excludes rows with actual consumption below 30 Wh (2.34% of train, 0.58% of validation and 0.38% of test rows).
- **Reasoning.** MAE measures typical error; RMSE weights large misses. Requiring both stops whichever metric favours a model from deciding the verdict alone.
- **Trade-offs.** The 30 Wh cutoff is a numerical safeguard, not a physical idle threshold, and its exclusion rate differs by partition.
- **Consequences.** The main result is a split: deep models win on MAE and lose on RMSE, so they are "not shown" better.

### ADR-007 · Pre-register evaluation rules; use the test partition once

*Phase 4 · Accepted*

- **Context.** Many configurations evaluated against a small held-out set invite selection bias and after-the-fact rule changes.
- **Options considered.** Informal comparison, or written rules committed before any result exists.
- **Decision.** Protocols were committed before the code that runs them. Phase 4 Tier 1 rule: the LSTM is "shown" better only if **each** of seeds 42, 43 and 44 has test MAE ≤ 0.99 × linear regression's test MAE **and** test RMSE ≤ 0.99 × its test RMSE. A good mean cannot rescue a failing seed. The test partition was scored once per seed and horizon: 6 LSTM evaluations in total, never repeated. Any rule change is a new dated entry that states how many test evaluations already exist.
- **Reasoning.** Verdicts can be verified against dated commits, and "not shown" becomes a legitimate outcome rather than something to tune away.
- **Trade-offs.** Slower iteration. A failed criterion cannot be revisited on the same data.
- **Consequences.** Tier 1 was **not shown**: all 3 seeds beat the MAE bar and all 3 missed the RMSE bar (`results/final_evaluation.json`). Comparing serving forecasts with real test-period values would count as a new test evaluation.

### ADR-008 · A structurally validation-only tuning path

*Phase 4 · Accepted*

- **Context.** The general harness scores train, validation *and* test on every call. Using it for tuning would load test labels and hold test predictions in memory.
- **Options considered.** (1) Reuse the harness and rely on discipline. (2) A separate evaluator and runner that are only ever given training and validation data.
- **Decision.** Option 2: `src/evaluation/validation_only.py` and `src/tuning/runner.py` take explicit train and validation file paths. Tests check that the tuning code contains no test path, no harness import and no MLflow logging.
- **Reasoning.** An isolation test can *demonstrate* the rule instead of assuming it.
- **Trade-offs.** About ten lines of prediction logic are duplicated from the harness. Equivalence tests against the harness guard against drift.
- **Consequences.** Tuning results (`results/tuning_h6_validation.json`) contain validation metrics only.

### ADR-009 · Walk-forward comparison design

*Phase 5 · Accepted*

- **Context.** One validation window is too thin to compare model families, and the test partition is spent (ADR-007).
- **Options considered.** Sliding or expanding windows; choosing a "best classical" reference after seeing results, or fixing references in advance; weighted or unweighted aggregation.
- **Decision.** 8 expanding-window folds with 7-day evaluation windows (1,008 readings each) starting 2016-03-01, all at h = 6. Each fold refits its train-only scaler and purges the last 6 training labels. Linear regression and random forest are fixed references. For each seed, the per-fold ratio (model ÷ reference) is averaged over 8 folds. **Shown better:** both MAE and RMSE mean ratios ≤ 0.99 for every seed. **Shown worse:** both ≥ 1.01. **Not shown:** anything else. The rule constants are in `config/walk_forward.py`, and the results writer refuses to overwrite.
- **Reasoning.** Expanding windows match periodic retraining on all history. Fixing references in advance avoids picking a weak opponent afterwards. Equal-length folds justify an unweighted mean.
- **Trade-offs.** Folds are consecutive slices of one 138-day period, not independent samples. Folds 7–8 overlap the validation period used to tune the LSTM; a folds 1–6 sensitivity check is reported alongside.
- **Consequences.** All 6 comparisons are "not shown": deep models' MAE ratios are ≤ 0.99 for every seed, but their RMSE ratios are ≥ 1.01. The folds 1–6 sensitivity check gives the same classification.

---

## Tracking & artifacts

### ADR-010 · One MLflow store, aliases not stages, tracking outside the core

*Phase 0–3 · Accepted*

- **Context.** Notebooks relied on MLflow's default tracking URI, which resolves relative to the working directory. This silently created three disconnected stores.
- **Options considered.** Per-notebook configuration, or one importable configuration module.
- **Decision.** `config/mlflow_config.py` defines one absolute SQLite URI (`db/mlflow.db`) that every caller sets explicitly. The registry uses the `@champion` alias, not the deprecated stages API. Logging is a thin consumer of an evaluation result (`src/tracking/mlflow_logger.py`); models, the harness and serving never import MLflow.
- **Reasoning.** One source of truth prevents fragmentation. A regression test checks that isolated test runs never create or modify the real store.
- **Trade-offs.** The store is local and git-ignored, so it is not shared.
- **Consequences.** MLflow is a training-time tool only; deployment uses exported bundles (ADR-016).

### ADR-011 · Serialize models without pickle

*Phase 3–6 · Accepted*

- **Context.** Serving must load fitted estimators, scalers and neural networks, and pickle can execute arbitrary code on load.
- **Options considered.** pickle/joblib; skops for sklearn objects; a full `torch.save` of model objects, or state dicts only.
- **Decision.** sklearn estimators and scalers use skops, with untrusted types allow-listed per family: only `sklearn.tree._tree.Tree`, only for random forest. Neural networks save only their `state_dict`, loaded with `weights_only=True` into a network rebuilt from the parameters recorded in `schema.json`. Each bundle's `schema.json` pins the feature contract, and loading fails if it differs from `config/features.py`. skops files are ZIP-compressed.
- **Reasoning.** Loading needs no code execution, and a bundle cannot silently disagree with the current feature pipeline.
- **Trade-offs.** One loader per family, and skops' trust review is extra work. Importing skops pulls in `torch` transitively, so the original "no torch in the process" claim for linear-regression serving was withdrawn and replaced by tests showing no torch computation on that path.
- **Consequences.** Compression shrinks the random-forest estimator from 89 MB to 15 MB, which makes ADR-016 practical.

---

## Serving

### ADR-012 · Linear regression as primary serving model; all five families served

*Phase 6 · Accepted (post-hoc)*

- **Context.** No deep model was "shown" better (ADR-009), yet a service needs a default model.
- **Options considered.** Serve the lowest-MAE model (GRU); serve the lowest-RMSE model (linear regression); serve several.
- **Decision.** Linear regression is the primary model, chosen after the walk-forward results with RMSE as the criterion, because large errors matter more for point energy forecasts. Random forest, LSTM, GRU and CNN-LSTM were then trained fresh on the final window with seed 42 and the same fixed configuration, and all five are served side by side. These are new runs, not the walk-forward models.
- **Reasoning.** Mean walk-forward RMSE: linear regression 87.40, random forest 87.71, GRU 90.50, CNN-LSTM 90.59, LSTM 90.73. A wording rule fixed before the per-fold table was examined requires that the edge over random forest *not* be described as a win: it rests on fold 1, and removing that fold flips its sign. Against the deep models, linear regression has the lower RMSE in 8/8 (LSTM), 7/8 (GRU) and 6/8 (CNN-LSTM) folds. The remaining tie-breakers are operational and were not measured: a smaller artifact and no tree-type trust exception.
- **Trade-offs.** The choice is post-hoc and RMSE-specific; on MAE the deep models are lower. Linear regression uses squared loss and the deep models use Huber loss, so the objectives differ as well as the model classes.
- **Consequences.** `/predict` takes a list of families. The app shows forecasts unranked, with the "not shown" caveat.

### ADR-013 · Final models train on train + validation, never on test

*Phase 6 · Accepted*

- **Context.** Deployed models should use all non-test history without touching the held-out period.
- **Options considered.** Reuse the fold builder with an evaluation window over the test period, or compose a training-only window.
- **Decision.** `src/training/final_window.py` trims raw rows to before 2016-04-30 *before* building features, applies the standard h = 6 purge and fits the scaler on the final-window feature frame. This gives 15,738 rows under the date mask, 15,594 with a finite daily lag, and 15,588 fittable rows.
- **Reasoning.** "Test untouched" holds literally: no test row can reach a feature, target or scaler.
- **Trade-offs.** The serving scaler differs from the one used in the offline evaluations (it has seen the validation period).
- **Consequences.** Every bundle records its training window, row count, code SHA and scaler convention in `schema.json`.

### ADR-014 · Stateful serving with a seeded rolling buffer

*Phase 6 · Accepted*

- **Context.** Features need up to 144 readings of history, and sequence models need 17 more rows of context.
- **Options considered.** Stateless requests carrying full history (burden on the client), or a server-side buffer.
- **Decision.** A server-side buffer of 162 readings (145 for one feature row + 17 for an 18-row window). At startup it is seeded from stored history ending **2016-04-29 23:50**, the last pre-test timestamp. `POST /ingest` appends a reading, which must be exactly 10 minutes after the last one (409 for duplicates or gaps, 422 for timezone-aware or non-finite input). `POST /predict` is read-only, takes a list of families, and reports an unknown family as `not_available` without failing the whole request. Features are built by the same `build_features` code used offline, and columns are selected by name.
- **Reasoning.** Clients send only the new reading, restarts are deterministic, and the server never fabricates or shifts data. Separating ingest from predict lets several models score the identical state.
- **Trade-offs.** State is in memory: restarting resets the buffer to the seed. The 162-row size assumes L = 18 and must be recomputed if that changes.
- **Consequences.** The same `ServingService` backs the API and the Streamlit app (ADR-017).

### ADR-015 · Pre-registered serving-equivalence bars

*Phase 6 · Accepted, amended once*

- **Context.** Serving must produce the same features and predictions as the offline pipeline.
- **Options considered.** Spot checks, or numeric bars fixed before running the comparison.
- **Decision.** `|a − b| ≤ max(1e-12·|b|, floor)`, with a 1e-12 Wh floor for raw features and 1e-10 Wh for predictions, checked on pre-test rows only (`tests/test_serving_equivalence.py`).
- **Reasoning.** Fixing tolerances in advance prevents fitting them to observed differences.
- **Trade-offs.** The first run failed on `roll6_std` (4.06e-11) and `roll18_std` (1.68e-11): rolling variance accumulates differently over a 162-row buffer than over the full series. The relative bar was raised to 1e-10 **for those two columns only**. All other bars, including the prediction bar, are unchanged and pass.
- **Consequences.** Train/serve skew is covered by a test rather than by convention.

---

## Deployment

### ADR-016 · Ship exported model bundles in the repository

*Public release · Accepted*

- **Context.** The registered models lived only in the local, git-ignored MLflow store (SQLite plus 241 MB of run artifacts with absolute paths), so a fresh clone or a Streamlit Community Cloud deployment could not load any model. The random-forest bundle alone was 89 MB.
- **Options considered.** (1) Commit the MLflow store: too large, and its paths are machine-specific. (2) A hosted MLflow server or object store: needs infrastructure and secrets, which is excessive for a demo. (3) Git LFS: extra tooling, and hosting support is uncertain. (4) Export each `@champion` bundle plus the 162-row seed history into `models/`.
- **Decision.** Option 4. `scripts/export_serving_models.py` writes `models/<family>/` (schema, scaler, estimator or state dict) and `models/seed_history.csv`, about 15 MB in total. Before writing, it checks that the exported bundles give bit-identical predictions to the registry bundles.
- **Reasoning.** It is the smallest change that makes the app and the API run from a clone with no services, secrets or data download.
- **Trade-offs.** 15 MB of binaries in git, and a re-registered model must be re-exported. Changes to `models/` deserve the same review as code, because the files are deserialized.
- **Consequences.** Runtime dependencies drop MLflow and DVC. Serving tests run against the committed bundles on any machine.

### ADR-017 · Streamlit runs inference in-process with per-session state

*Public release · Accepted*

- **Context.** The Live page used to start the FastAPI app as a uvicorn subprocess on a free port and call it over HTTP. The single shared buffer meant that concurrent public visitors would reject each other's readings (409s), and the setup added process management and three dependencies to the hosted app.
- **Options considered.** (1) Keep the subprocess. (2) Deploy the API and the UI separately. (3) Call the serving core directly.
- **Decision.** Option 3. Bundles load once per server process (`st.cache_resource`); each browser session gets its own `ServingService` with a freshly seeded buffer (`st.session_state`). A process-wide lock serializes predictions, because sequence models set torch's global thread count. The FastAPI app remains a separate entry point built on the same `create_service` factory.
- **Reasoning.** Visitors are isolated, there are no ports or child processes, and the app works on any Streamlit host.
- **Trade-offs.** The app no longer exercises the HTTP layer; the API has its own tests. Predictions are serialized across sessions (each takes milliseconds).
- **Consequences.** `streamlit run streamlit_app.py` is the whole deployment. `tests/test_streamlit_app.py` renders every page and adds a reading through the real widgets.

### ADR-018 · Reproducible data from the public source

*Public release · Accepted*

- **Context.** DVC tracked the data, but its only remote was a directory on the author's machine, recorded in the committed `.dvc/config`. The local CSV is byte-identical to UCI's `energydata_complete.csv` (MD5 `69ef922b5fcafcd49097cfc09e07167e`, CC BY 4.0), and `src.build_processed` regenerates the six processed CSVs byte-for-byte.
- **Options considered.** Publish a DVC remote; commit the data; download from UCI with verification.
- **Decision.** `scripts/download_data.py` downloads the archive and writes the CSV only if the checksum matches (`make data` also rebuilds `data/processed/`). The `.dvc` files stay as hash records, and the remote moved to the git-ignored `.dvc/config.local`. Tests that need data are marked (`requires_raw_data`, `requires_processed_data`, `requires_registry`) and skip when it is absent.
- **Reasoning.** Anyone can reproduce the exact inputs from the original source without access to private storage, and the maintainer's DVC workflow keeps working.
- **Trade-offs.** Depends on UCI hosting the file (the checksum catches silent changes). `scaler_train_fit.joblib` and `split_boundaries.json` in the DVC-tracked directory are written only by notebook 02, not by `make data`; no library code reads them.
- **Consequences.** A fresh clone passes the full test suite with data-dependent tests skipped; after `make data`, the full suite runs.
