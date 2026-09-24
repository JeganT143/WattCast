# WattCast — Phases 0–2 Summary

## Phase 0 — MLOps Foundation

- **DVC** tracks raw and processed data (`data/raw/energy_data_set.csv.dvc`, `data/processed.dvc`), backed by a local remote (`.dvc/config`: `remote = local`, pointing at an out-of-repo cache directory) — keeps large/binary data artifacts out of Git history while still version-controlling exactly which data produced which result.
- **MLflow** tracks experiments against a SQLite store at `db/mlflow.db`, with model artifacts under `models/` and the `WattCast` experiment name — all defined once in `config/mlflow_config.py` (`TRACKING_URI`, `ARTIFACT_ROOT`, `EXPERIMENT_NAME`), built from an absolute `PROJECT_ROOT` so the store resolves identically regardless of the importing process's working directory.
- **Model registry** uses the alias API (`@champion`) rather than the deprecated stages API (`Staging`/`Production`) — see Decision #1 in the Phase 1 log below.
- Established early: every module/notebook touching MLflow must import `TRACKING_URI` from `config/mlflow_config.py` and call `mlflow.set_tracking_uri()` explicitly — the cost of not doing this (silent store fragmentation) was later confirmed directly in Phase 3; see that section's "MLflow store fragmentation" incident.

## Phase 1 — EDA Key Findings

See the full "Phase 1: Deep EDA Summary" below for details. Headline findings:
- Dataset (UCI Appliances Energy Prediction): 19,735 rows, 29 columns, strict 10-minute cadence, zero nulls, zero duplicate/missing timestamps.
- Strong, stable-shaped daily seasonality (overnight trough, morning rise, evening peak); weekday/weekend profiles differ in shape, not just level.
- ACF shows persistent daily dependence (lag 144 and multiples); PACF collapses to ~0 by lag 2–3 — `lag_144` kept as an empirical candidate regardless, since PACF only tests linear redundancy.
- No evidence extreme values are sensor errors — raw target retained unmodified, no clipping.
- STL (period=144, `robust=True`): daily rhythm *shape* is stable across the 4.5 months; *amplitude* declines monotonically (Early→Middle→Late).
- ADF test decisively rejects a unit root (statistic −21.62) — reconciled with the STL trend finding as bounded/mean-reverting movement, not unbounded stochastic drift.
- All same-timestamp sensor correlations with `Appliances` are weak (< |0.25|) — real predictive structure is temporal (daily rhythm, recent history), not raw contemporaneous sensor values.
- **Forecast framing (governs all of Phase 2):** true forecasting, not nowcasting — any feature for target `Appliances(t+h)` must be available at or before forecast origin `t`. Contemporaneous sensor readings at `t` are not usable raw; only their lagged/historical values are. Calendar-derived features are always usable regardless of horizon.

## Phase 2 — Feature Engineering

- **Lag features** (`src/features/lag.py`): `shift(N)` with strictly positive `N` for `LAG_STEPS = [1, 2, 3, 4, 5, 6, 144]` (config/features.py) — pulls only past values into row `t`. Rejects `N <= 0` (that would be a target, not a feature).
- **Rolling features** (`src/features/rolling.py`): backward-only rolling mean/std over `ROLLING_WINDOWS = [6, 18]`, `center=False` always — a centered window is undefined at serving time since a live context buffer has no future rows.
- **Cyclical/calendar features** (`src/features/temporal.py`): `hour_of_day`, `day_of_week`, `is_weekend` (raw, for tree models) plus sin/cos encodings of minute-of-day (period 1440, matching the true 10-minute resolution) and day-of-week (period 7). Carry no leakage risk — calendar position at any `t` is always knowable in advance.
- **Targets** (`src/features/targets.py`): mirror image of lag — `shift(-N)` for each horizon in `TARGET_HORIZONS = [1, 6]`.
- **Leakage-safety rule, enforced structurally, not just by convention**: features only ever look backward (`lag.py`/`rolling.py` reject non-positive shift/window arguments); targets only ever look forward (`targets.py` rejects non-positive horizons). The boundary that matters is features-vs-target, not past-vs-future in isolation.
- **Time-aware split** (`src/data/split.py`): chronological train/val/test boundaries (`SPLIT_TRAIN_END`, `SPLIT_VAL_END` in `config/features.py`) as boolean masks, not separate dataframes — feature construction runs on the full continuous series *first*, so lag/rolling features retain temporal context across split boundaries (e.g. val's first row can still see train's tail for `lag_144`).
- **Scaling** (`src/preprocessing/scaling.py`): `StandardScaler` fit only on train-partition rows (`fit_scaler`), then applied unchanged to val/test/serving rows (`transform_with_scaler`) — prevents val/test distribution statistics from leaking into the fitted scaler.
- **Single source of truth**: `config/features.py` centralizes `LAG_STEPS`, `ROLLING_WINDOWS`, `SEASONAL_PERIOD`, `TARGET_HORIZONS`, `FEATURE_COLUMNS`, `SCALED_COLUMNS`, and split boundaries. `src/features/build_features.py` and `src/pipeline.py` orchestrate the above into one canonical sequence that both training and serving must call — never reimplemented independently, to avoid train/serve skew.

---

# WattCast — Phase 1: Deep EDA Summary

**Dataset:** UCI Appliances Energy Prediction (Candanedo et al.) — 19,735 rows, 29 columns, 10-minute intervals, ~4.5 months (2016-01-11 to 2016-05-27).

---

## 1. Structural & Timestamp Integrity

- Loaded raw CSV; `date` column initially parsed as `object` (string) — converted to `datetime64` and set as the DataFrame index.
- **Zero duplicate timestamps.** Strictly monotonic increasing.
- **Zero cadence violations** — every consecutive gap is exactly 10 minutes across all 19,734 intervals.
- **Zero nulls** across all 29 columns.
- Caught and fixed a boundary bug: `.diff()`'s first-row `NaT` was initially inflating the "cadence violation" count by one — fixed by explicitly excluding `NaT` from the comparison (`time_diff.notna() & (time_diff != expected_gap)`).
- **Conclusion:** dataset is likely pre-cleaned by the original authors (real sensor networks rarely have zero dropouts over 4.5 months); timestamps are fully trustworthy, so rolling/lag features in Phase 2 don't need gap-aware logic.
- Note: `rv1` and `rv2` are deliberately injected random-noise columns from the original dataset authors, used later as a validity check for correlation and eventually feature-importance methods.

## 2. Daily & Weekly Seasonality

- **Hourly profile:** clear non-random daily shape — overnight trough (~48–55, 00:00–05:00), sharp morning rise (06:00–08:00), continued climb to a mid-morning local peak (~133 at 11:00), a midday plateau/dip (11:00–15:00), a strong second rise into a real evening peak (~190 at 18:00), then steady decline back to overnight levels.
- **Weekday vs. weekend:** the two profiles are *not* a constant vertical shift — the shape itself changes. Weekend consumption is substantially higher through late morning/early afternoon (occupants home longer); weekday evening peak is slightly higher and more concentrated (occupants returning home at a specific time). Overnight hours are nearly identical regardless of day type.

## 3. ACF / PACF

- **ACF:** sharp decay from lag 1, near-zero by ~lag 40–50 (7–8 hrs); a real secondary bump at ~lag 144 (24 hrs), repeating at every subsequent daily multiple out to lag 1008 (7 days) with roughly constant height — i.e., the daily cycle persists undiminished across the full week, rather than providing independent evidence of a distinct weekly effect (that evidence came from the weekday/weekend grouped means instead, not from ACF).
- **PACF:** collapses to ~0 by lag 2–3, including at lag 144 — meaning lag-144 adds no *independent linear* information beyond the short lags already occurring in the chain (144 = 143 short-range links + relative daily-cycle position).
- **Decision:** build `lag_144` as an empirical candidate anyway, not a foregone inclusion or exclusion. PACF only tests linear redundancy via an autoregressive framework; a nonlinear model (Random Forest, LSTM) may still find real signal in it that PACF cannot detect. `lag_144` encodes *realized historical behavior*; `hour_sin`/`hour_cos` encode *calendar position* — genuinely different, complementary information.

## 4. Outlier Investigation

- Rolling (backward-looking) IQR method (1-day window) flagged **11.26%** of observations — far too high a rate to represent genuine rare anomalies; diagnosis: the rule was capturing the series' normal bursty variability, not errors.
- Manually inspected extreme values (max = 1080, several other readings > 800–900) and both sustained multi-hour high-usage runs and 99.5th-percentile-isolated points: every case sits inside a smooth, physically plausible local ramp-up/ramp-down — no context-free spikes, no stuck/flat sensor plateaus, no physically impossible values (no negatives, etc.).
- **Decision:** retain the raw target unmodified. No clipping, no removal. High-consumption events are treated as real, important signal the model must learn — not noise to clean away. The investigation's value was establishing *evidence* for this decision, not merely finding nothing wrong.

## 5. STL Decomposition (Trend / Seasonal / Residual)

- Period = 144 (daily cycle, confirmed via ACF). `robust=True` selected: `robust=False` allowed individual extreme (but legitimate) events to visibly distort the trend estimate (swinging on a day-to-day timescale instead of a genuine slow-moving level); `robust=True` produced a materially smoother, more interpretable trend.
- Raw seasonal `max − min` amplitude (both full-series and naive per-day) proved fragile — dominated by a handful of extreme days/timestamps (same failure mode as the outlier investigation). Replaced with a robust per-day **P90−P10** metric.
- **Finding:** daily seasonal amplitude declines **monotonically** across the study window — Early (median 94) → Middle (74) → Late (60) — plausibly tracking winter→spring daylight/temperature change (hypothesis noted, not yet formally cross-checked against `T_out`).
- **Overall conclusion:** the daily rhythm's *shape* is stable across all 4.5 months (same overnight-low → morning-rise → evening-peak structure throughout), but its *amplitude* is not stable — it drifts over the study period. This is a genuinely different, more precise finding than either "seasonality is stable" or "seasonality is unstable" alone.

## 6. Stationarity (ADF Test)

- `adfuller(df["Appliances"], regression="ct", autolag="AIC")` → statistic = **−21.62**, p ≈ **0.0**, decisively below even the 1% critical value (−3.96).
- **Conclusion:** strong rejection of the unit-root null — no evidence of unit-root-type non-stationarity.
- **Reconciled with the STL trend finding:** these are different questions. STL's trend describes *bounded, mean-reverting local-level movement* (the series reliably swings back down after every peak, driven by strong short-lag autocorrelation and daily mean-reversion). ADF's unit-root question is about *unbounded stochastic drift* — a series that could wander arbitrarily far from any fixed level given enough time. The Appliances series does the former, not the latter.
- Baselines (Linear Regression, Random Forest) don't require stationarity; no differencing applied to the target. The test was still worth running for the intuition/evidence it produced, not because the baselines needed it.

## 7. Sensor Correlation

- Pearson correlation of all numeric features against `Appliances` (same-timestamp).
- All correlations are weak — strongest are `RH_out` (−0.15) and `time_of_day` (+0.22); every sensor sits under |0.25|.
- **Sanity check passed:** `rv1` and `rv2` (deliberately injected noise columns) show near-zero correlation (−0.011 each), as expected — confirms the correlation methodology is behaving correctly.
- **Interpretation:** no individual sensor linearly explains much of the target on its own — the real predictive structure is temporal/dynamic (daily rhythm, recent history), not simple contemporaneous sensor relationships. This directly supports prioritizing time-based and lag features over raw sensor values alone.

## 8. Forecast Framing Decision (critical for Phase 2)

- WattCast's architecture (sliding-window context buffer, sequence models) indicates **true forecasting**, not nowcasting: predicting `Appliances(t+h)` using only information available at or before the forecast origin.
- **Governing rule for Phase 2:** any feature must be available at or before the moment the forecast is generated.
  - Contemporaneous sensor readings at the *target* timestamp (including `lights`, and every `T*`/`RH*` column) are **not** usable in their raw same-timestamp form — only their lagged/historical values are.
  - Calendar-derived features (`hour_sin`/`hour_cos`, day-of-week, etc.) for the *target* timestamp **are** usable, since calendar position is knowable in advance regardless of horizon.
- The Phase 1 correlation results remain valid as EDA (informing which sensors are worth lagging into features later) but must not be read as license to use those same-timestamp values directly.

---

## Phase 1 Synthesis (one paragraph)

The dataset has regular 10-minute timestamps with no meaningful timestamp integrity problems, so the temporal ordering is suitable for time-series modeling. The target shows a strong and persistent daily cycle — low overnight, a morning rise, and a major evening peak — with weekday and weekend profiles differing meaningfully, particularly during daytime hours. The ACF confirms recurring daily dependence; the evidence does not independently justify a distinct weekly cycle claim from ACF alone (that came from the grouped weekday/weekend means). The extreme-value investigation found no evidence that large consumption values are sensor errors, so the target is retained unmodified. STL shows the broad daily rhythm persists throughout the 4.5 months but its amplitude and fine shape drift over time — most plausibly a seasonal/daylight effect. The ADF test strongly rejects a unit root despite this local drift, because STL's trend describes bounded, mean-reverting movement rather than unbounded stochastic drift. Same-time sensor correlations with the target are uniformly weak (confirmed via the `rv1`/`rv2` noise-column sanity check), meaning the real predictive structure lies in temporal patterns and recent history rather than raw contemporaneous sensor values. Because WattCast is a true forecasting system, only information available at or before the forecast origin may be used as a feature for the target timestamp — a rule that governs every design decision in Phase 2.

---

## Time-Series Methodology Decisions Log (Phase 1)

| # | Decision | Alternative(s) Considered | Reason |
|---|----------|---------------------------|--------|
| 1 | MLflow registry via aliases (`@champion`) | Deprecated stages API (`Staging`/`Production`) | Aliases are the current, non-deprecated pattern |
| 2 | Timestamp cadence confirmed uniform, no gaps | — | Verified via `.diff()` check (with boundary-NaT fix) |
| 3 | `lag_144` kept as an empirical candidate feature | Discard based on PACF≈0 at lag 144 | PACF only tests linear redundancy; nonlinear models may still benefit |
| 4 | Raw `Appliances` target retained unmodified (no clipping/removal) | Rolling-IQR auto-removal; hard percentile clipping | 11.26% flagged is too high to be "rare anomalies"; inspected extremes show smooth, plausible ramps, no sensor-error evidence |
| 5 | STL `robust=True` | `robust=False` | Prevented individual extreme events from distorting the trend estimate |
| 6 | Seasonal amplitude measured via per-day P90−P10 | Raw `max − min` (full-series and per-day) | Max-min dominated by edge effects and a few extreme days; P90-P10 is resistant to both |
| 7 | No differencing applied to target despite STL trend | Differencing to enforce stationarity | ADF strongly rejects unit root; baselines (LinReg, RF) don't require stationarity |
| 8 | WattCast framed as true forecasting, not nowcasting | Nowcasting (contemporaneous sensors allowed) | Architecture uses a rolling context window of past timesteps; governs which features are leakage-free in Phase 2 |

---

## Failure Modes Triggered & Understood

1. **Boundary `NaT` in `.diff()`** silently inflating a cadence-violation count — caught and fixed.
2. **Fragile max-min-based summary statistics** (outlier detection, seasonal amplitude) distorted by a small number of extreme points/edge effects — caught twice (rolling-IQR outlier flag rate, then STL seasonal amplitude at both full-series and per-day granularity) and replaced with percentile-based, outlier-resistant metrics both times.
3. **(Caught before it could occur) Contemporaneous-feature leakage** — initially spotted via the `lights` column, then correctly generalized to *every* contemporaneous sensor reading once the true-forecasting framing was established.

---

*Phase 1 complete. Proceeding to Phase 2: leakage-checked feature engineering, beginning with defining the forecast horizon.*


# WattCast — Decisions Log

## Phase 3 — Forecaster Interface, Baselines, Evaluation Harness, MLflow Integration

### Architecture decisions

- **`Forecaster` interface** (`src/models/forecaster.py`): `fit(X, y) -> Forecaster`, `predict(X) -> np.ndarray` (always 1-D), `params -> dict`, `required_columns -> list[str]`. Locked before any model implementation, per the project's Strategy-pattern requirement. `fit()` returning `self` is uniform across every implementation, including no-op baselines — a no-op `fit()` is a legitimate contract fulfillment, not a violation, since `fit()` means "prepare for prediction using training data," not "every implementation must learn parameters."
- **`required_columns` as a `Forecaster` property**, not a shared global — added after discovering the naive persistence baseline needs `Appliances[t]` itself, which is deliberately excluded from `FEATURE_COLUMNS` (a decision made in Phase 2 to keep "current state" represented only through lag/rolling features for learned models). Each concrete forecaster declares its own input contract; the harness builds `X = df[forecaster.required_columns].to_numpy()` generically, never branching on model type.
- **Context columns generalized in `assemble.py`**: `build_horizon_dataset()` now accepts `context_columns` (default `[]`), unioned into both column selection and `dropna` scoping alongside `feature_columns` + `target_column`. Used for `Appliances` (both horizons) and horizon-specific seasonal-naive lags (`Appliances_lag_143` for t+1, `Appliances_lag_138` for t+6).
- **`BASELINE_CONTEXT_LAGS` derived, not hardcoded**: `seasonal_lag(horizon) = SEASONAL_PERIOD - horizon`, computed by a validated helper (`0 < horizon < SEASONAL_PERIOD`) at import time in `config/features.py`. Prevents the seasonal-context lags from silently going stale if `TARGET_HORIZONS` changes.
- **Evaluation harness** (`src/evaluation/harness.py`): `evaluate_forecaster()` fits once on train, predicts on train/val/test, returns one `EvaluationResult` (dataclass, validated in `__post_init__`: exact partition keys, exact metric keys, 1-D shape-matched predictions/actuals). Deliberately single-window — walk-forward (Phase 5) will wrap this function repeatedly rather than duplicating its logic.
- **MLflow logging kept strictly outside the core architecture**: `Forecaster`, model wrappers, and `evaluate_forecaster()` have zero MLflow awareness. `src/tracking/mlflow_logger.py` is a thin consumer of an already-computed `EvaluationResult`. `native_model: Any | None = None` is passed explicitly by the caller (not via an added `get_native_model()` interface method) — keeps the `Forecaster` ABC unchanged and avoids the logger ever branching on concrete type.
- **Adapter pattern for sklearn models** (`src/models/sklearn_models.py`): `LinearRegressionForecaster`, `RandomForestForecaster`. No scaling or feature engineering inside either wrapper — scaling already happened in the Phase 2 pipeline; re-scaling here would risk train/serve skew.

### MAPE threshold: 30.0 Wh

Chosen empirically after inspecting the real `Appliances` distribution across train/val/test (Phase 2 pipeline output):
- `Appliances` never reaches exactly 0 in this dataset (observed floor: 10 Wh train, 20 Wh val/test); median is 60 Wh across all three partitions.
- 30 Wh is a **conservative, numerically-motivated cutoff** to avoid unstable/disproportionate percentage errors near the lowest observed values — **not** a claim that 30 Wh is a scientifically established idle boundary.
- **Known limitation, documented rather than hidden**: exclusion rate is asymmetric across partitions due to the dataset's non-stationary amplitude (Phase 1 STL finding). At threshold 30: train excludes 2.34%, val excludes 0.58%, test excludes 0.38% of observations.
- MAE and RMSE are computed over **all** observations in every case; only MAPE applies this threshold.

### Diagnostic: why does naive persistence have higher train MAPE than val/test?

Investigated rather than assumed, using a full error-distribution breakdown (per-partition absolute error stats, MAPE-used row counts, target-value quantiles among MAPE-used rows, relative-error stats):

- **Median absolute error is 10 Wh across train/val/test** — the typical prediction is equally accurate in every partition.
- **Train has a heavier tail of large persistence errors** (mean abs. error 31.30 in train vs. 25.26/26.52 in val/test; relative-error std 43.50 in train vs. 32.95/34.28 in val/test), which pulls train's mean-based metrics (MAPE, MAE) up more than val/test's.
- **MAPE-used target distributions have broadly similar lower quantiles** across partitions (5th/25th/50th percentiles nearly identical) — ruling out "train's denominators are just smaller" as the explanation.
- **Not claimed**: that the Phase 1 amplitude-decline finding caused this difference — that causal link was not established by this diagnostic. The larger train sample providing more opportunity to observe rare large-error events is noted only as a plausible contributing factor, not a demonstrated cause.

### Empirical results — real Phase 2 data, all 4 baseline models, both horizons

| Model | Horizon | Train MAE | Val MAE | Test MAE | vs. persistence (test) |
|---|---|---|---|---|---|
| naive_persistence | 1 | 30.84 | 25.24 | 26.50 | — |
| naive_seasonal | 1 | 68.09 | 48.75 | 53.72 | worse |
| linear_regression | 1 | 31.54 | 26.16 | 27.16 | worse (+2.5%) |
| random_forest | 1 | 12.72 | 29.30 | 32.41 | worse (+22.3%) |
| naive_persistence | 6 | 60.27 | 49.16 | 47.60 | — |
| naive_seasonal | 6 | 68.02 | 48.75 | 53.78 | worse |
| linear_regression | 6 | 51.10 | 44.41 | 42.39 | **better (−10.9%)** |
| random_forest | 6 | 15.26 | 42.34 | 45.20 | **better (−5.0%)** |

**Finding 1 — naive persistence beats naive seasonal at both horizons, decisively.** Test MAE at h=1: 26.50 (persistence) vs. 53.72 (seasonal) — roughly half the error. Consistent with Phase 1's ACF/PACF finding (sharp short-lag autocorrelation decay) combined with the STL finding of declining daily amplitude: recent activity is a stronger, more immediate signal than "same time yesterday" in a household whose overall energy behavior is drifting over the observation window.

**Finding 2 — at h=1, no learned model beats naive persistence.** Both Linear Regression and Random Forest underperform the naive baseline on val/test. Mechanistically consistent with h=1 being dominated by short-lag autocorrelation, which persistence captures directly; a learned model built on the same lag features has no obvious mechanism to beat that at a 10-minute horizon. This is treated as a legitimate, well-supported result per the project's evaluation protocol, not a failure to explain away.

**Finding 3 — at h=6, both learned models beat naive persistence**, where persistence's "hold flat" assumption has more room to be wrong over a longer horizon and the learned models' access to multiple lags/rolling stats/calendar features starts to pay off.

**Finding 4 — Random Forest shows clear overfitting at default hyperparameters (`n_estimators=100, max_depth=None`)**: train MAE is 2.3–2.8x smaller than val/test MAE at both horizons (e.g. h=1: train 12.72 vs. val 29.30). Linear Regression shows no such gap. RF's h=6 win over persistence is happening *despite* this overfitting, not because of a clean generalizable signal — a legitimate hypothesis for hyperparameter tuning (e.g. capped `max_depth`) in a later pass, not addressed in this baseline run per the "no tuning yet" scope of Phase 3.

### MLflow store fragmentation — incident and fix

During this phase, three separate, disconnected MLflow SQLite stores were discovered to exist simultaneously: the intended `db/mlflow.db` (2 legitimate Phase 0 verification runs), a stray repo-root `mlflow.db` (10 disposable runs, empty params/metrics/tags — confirmed via full inspection before any deletion), and a stray `notebooks/mlflow.db` (0 runs). Root cause: no shared, importable config module existed for the tracking URI; different notebook sessions relied on MLflow's implicit default, which resolves relative to whatever directory the process happens to run from.

**Fix**: created `config/mlflow_config.py` with `TRACKING_URI`/`ARTIFACT_ROOT`/`EXPERIMENT_NAME` built from an absolute `PROJECT_ROOT`, mirroring `config/paths.py`'s existing pattern. Every module/notebook touching MLflow must import from here and call `mlflow.set_tracking_uri(TRACKING_URI)` explicitly — never rely on the default. Stray stores (repo-root `mlflow.db`, `mlartifacts/`, `notebooks/mlflow.db`) deleted after confirming via direct inspection (run IDs, params, metrics, tags, timestamps) that all 10 stray runs were empty verification noise with no real content.

**Regression guard added**: `test_logging_never_touches_real_tracking_store` in `tests/test_mlflow_logger.py` asserts the real store's run count is unaffected by an isolated test run — directly guards against this exact failure mode recurring.

### MLflow version-specific finding (MLflow 3.16.1)

`mlflow.sklearn.log_model(..., artifact_path=...)` is deprecated in favor of `name=...`. More importantly: MLflow 3.x introduced **Logged Models** as a distinct tracking entity, separate from run artifacts — `client.list_artifacts(run_id)` does **not** surface a model logged via `mlflow.sklearn.log_model()`; the correct retrieval is `mlflow.search_logged_models(filter_string=f"source_run_id='{run_id}'")`. Discovered by direct verification after an initial test incorrectly passed for the wrong reason (checking an artifact path that was never going to contain the model regardless of whether logging succeeded).

### skops security boundary — Random Forest model logging

`mlflow.sklearn.log_model()` on a fitted `RandomForestForecaster` initially raised `UntrustedTypesFoundException` on `sklearn.tree._tree.Tree` — skops' default security policy refuses to deserialize tree-based model internals without explicit trust, since a malicious file could set out-of-bounds node indices. Fixed by passing `skops_trusted_types=["sklearn.tree._tree.Tree"]` explicitly — scoped to exactly the flagged type, not a blanket trust-everything override, per skops' own guidance. Legitimate here because every model this project logs is self-trained from our own pipeline, never loaded from an external/untrusted source. **Flagged for re-evaluation** if Phase 6's serving layer ever loads models from anywhere other than this project's own MLflow registry.

Gap in test coverage that let this slip through initially: the native-model logging test only exercised `LinearRegressionForecaster` (no tree structure, not affected), not `RandomForestForecaster`. A Random-Forest-specific test (`test_random_forest_native_model_logged_when_supplied`) was added to close this gap.

### Known follow-up items (not blocking, logged for later)

- Three test gaps from the baseline-context plumbing step: no test directly asserts `Appliances_lag_143`/`_138` are absent from `FEATURE_COLUMNS`/`SCALED_COLUMNS` themselves (only from the assembled dataset); the unscaled-context test samples one row per horizon rather than the full partition; no test explicitly proves the persistence and seasonal-naive context columns coexist correctly in the same assembled dataset without collision.
- `pipeline.py`'s six `build_horizon_dataset()` calls are hardcoded per-horizon (`1`, `6`) rather than derived from `TARGET_HORIZONS` in a loop — an intentional readability tradeoff, but means `TARGET_HORIZONS` is not a fully single source of truth end-to-end.
- Stray/dead files identified during this phase: `tests/load_mlflow_model.py`, `tests/set_mlflow_alias.py`, `tests/tests_features.py` (vs. real `tests/test_features.py`), `src/features.py` (vs. real `src/features/` package), `src/data_loader.py`. None broke anything at the time; `tests/test_mlflow.py` (a genuinely broken scratch script hardcoding a live `localhost:5000` server, blocking full-suite collection) was deleted during this phase. The remainder were removed in the subsequent cleanup pass — see repo history; 106 tests still pass post-cleanup.
- 15 total runs exist in the `WattCast` MLflow experiment where 8 are expected — the first full-leaderboard attempt logged 7 runs successfully before crashing on the Random Forest skops error; the corrected re-run produced all 8 complete. The 7 early duplicates are valid but redundant, distinguishable by earlier `start_time`; left in place rather than deleted, since MLflow supports multiple runs per configuration natively and no data integrity issue exists.
- Random Forest overfitting (train MAE 2.3–2.8x better than val/test) not addressed — baseline run only, no tuning performed per Phase 3 scope.

### What Phase 3 has established

- A working, tested, Liskov-substitutable `Forecaster` interface — proven across 4 genuinely different implementations (2 no-op deterministic baselines, 2 sklearn-backed learned models) with zero special-casing in the evaluation harness.
- A real, evidence-backed leaderboard for both forecast horizons, with a mechanistically-explained result (not just numbers): naive persistence is a strong baseline at h=1 that no untuned learned model beats; both learned models beat persistence at h=6; Random Forest's h=6 win is confounded by overfitting.
- A working MLflow integration, verified against the actual installed library version's real behavior (not assumed API), with 9 flattened metrics, params, git-commit traceability, prediction/actual artifacts, scaler artifacts, and native model artifacts, per run.
- A fixed, structurally-guarded MLflow tracking configuration, with a regression test preventing the exact fragmentation incident that occurred mid-phase from recurring.

### What remains open for Phase 4/5

- LSTM, GRU, CNN-LSTM implementations behind the same `Forecaster` interface (Adapter pattern, PyTorch this time).
- Walk-forward validation (currently single-window only) — a thin wrapper around `evaluate_forecaster()`, not yet built.
- Random Forest hyperparameter tuning (capped depth) — a concrete, evidence-motivated candidate for improvement, not yet attempted.
- Single-step vs. multi-step forecasting error accumulation comparison — not yet relevant until a model that feeds its own predictions forward exists.

## Phase 4: LSTM evaluation protocol (pre-registration), 2026-09-24

Registered BEFORE any LSTM test, implementation, or experiment exists. Repo HEAD at registration: 318c091.
Baseline reference: tests/fixtures/golden_baselines.json (last changed in c4c9583; purged data).

### Provenance of these decisions
- The residual diagnostic (h=6) printed test-split residuals as well as validation ones. Only validation statistics
  informed these decisions; test residuals were not used to choose anything below.
- Validation findings used (h=6): LR mean residual -4.00 but median -11.65 (squared-error fit sits above the median of
  a right-skewed target); persistence is bimodal (median |resid| 20, p90 170); only 45% of LR residuals are within 20 Wh.
- At h=1, LR is worse than persistence on MAE (+0.91 val, +0.66 test) and better on RMSE (-6.56, -6.43): the earlier
  claim "no learned model beats persistence at h=1" holds on MAE only.

### Primary question
- Primary horizon: h=6. Reference baseline: linear_regression at h=6 (lowest baseline test MAE and RMSE).
- h=1 is reported, not decision-making. It uses the selected hyperparameters below (no separate selection); its
  output bias uses the median of its own eligible target_t1 values. Seeds 42, 43, 44; one test evaluation per seed.

### Tier 1: success criterion (decision-making)
- Reference values are read UNROUNDED from the fixture in code: ref_mae and ref_rmse are the test_mae and test_rmse of
  "linear_regression|h6" (approx 42.3913 and 80.619).
- Reference separation: the Tier 1 thresholds use the TEST metrics of linear_regression|h6 (ref_mae, ref_rmse). The validation metrics of linear_regression|h6 are used only for hyperparameter selection (below). The two references are never mixed.
- All three seeds (42, 43, 44) must INDEPENDENTLY satisfy BOTH thresholds on the test partition:
  test_mae <= 0.99 * ref_mae and test_rmse <= 0.99 * ref_rmse (approx 41.967 and 79.813, informational only).
- Mean performance never rescues a failing seed. If any seed misses either threshold, the outcome is "not shown"; that
  is a valid, reportable result, not something to explain away or re-tune.

### Tier 2: reporting (not decision-making)
- Report each seed's test MAE and RMSE at h=6 and h=1, plus mean +/- sample standard deviation (ddof=1) across the
  three seeds, next to all Phase 3 baselines (naive_persistence, naive_seasonal, linear_regression, random_forest)
  from the fixture, at both horizons.
- No additional success criteria may be introduced after results are seen.

### Model, fixed (not tuned)
L=18 (required_history_length 17); LSTM hidden_size 64, num_layers 1, dropout 0; Linear(64, 1) head; Adam, learning
rate 1e-3, no weight decay, no LR schedule, no gradient clipping; batch_size 64; training windows shuffled each epoch
with a generator seeded from `seed`; float32 internally; target in raw Wh (no target scaling); loss computed in raw Wh.
Final output-layer bias initialised to the median of the y array passed to fit (eligible training targets), recorded
as a parameter (output_bias_init="train_target_median"). No early stopping; training runs exactly max_epochs epochs.
The seed controls weight initialisation and shuffle order. torch_num_threads = 4, fixed for every run.
Feature columns: FEATURE_COLUMNS minus hour_of_day and day_of_week (16 columns).

### Selection protocol (validation only)
- Grid: max_epochs in [10, 20, 35, 50] x huber_delta in [20, 40, 60] Wh (12 configurations), each fitted with seeds
  42, 43, 44 (36 fits), loss = Huber.
- Reference validation values, read UNROUNDED from the fixture entry "linear_regression|h6": lr_val_mae, lr_val_rmse
  (approx 44.416 and 81.931).
- Reference separation: hyperparameter selection uses the VALIDATION metrics of linear_regression|h6 (lr_val_mae, lr_val_rmse). The test metrics of linear_regression|h6 are used only for the Tier 1 thresholds (above). The two references are never mixed.
- For each configuration, mean_val_mae and mean_val_rmse are the arithmetic means of the per-seed validation metrics
  over seeds 42, 43, 44. Then:
    val_mae_ratio  = mean_val_mae  / lr_val_mae
    val_rmse_ratio = mean_val_rmse / lr_val_rmse
    selection_score = max(val_mae_ratio, val_rmse_ratio)
- Selected configuration = minimum of the key
  (selection_score, val_mae_ratio, val_rmse_ratio, max_epochs, huber_delta), compared lexicographically:
  lowest score; if exactly tied, lowest val_mae_ratio; then lowest val_rmse_ratio; then lower max_epochs;
  then lower huber_delta. No tolerance band.
- Test isolation: evaluate_forecaster computes test metrics as a side effect of every call. The tuning code must never
  read, persist, log, or use any test metric or test prediction, and must not call mlflow_logger. Only val_* fields
  are read; tuning output contains validation metrics only.
- The LSTM is scored on the full validation population (n_evaluated val = n_val), same as the baselines.

### Final evaluation
- The selected configuration is refitted with seeds 42, 43, 44 and evaluated on test exactly once per seed at h=6
  (and once per seed at h=1 with the same hyperparameters). Each pre-registered configuration receives one test
  evaluation. A run that crashes before producing any metric may be re-run once; log it. Nothing else may be re-run.
- The test evaluation is performed regardless of the selected configuration's validation score.
- The two-tier golden bar and the baselines are unchanged by this protocol.

### params (recorded for every run)
L, hidden_size, num_layers, dropout, learning_rate, batch_size, max_epochs, loss, huber_delta, output_bias_init,
seed, torch_num_threads, optimizer.

### Changing this protocol
Any change (grid, rule, seeds, thresholds, architecture, thread count) is a NEW dated and committed registration
that states how many test evaluations have already occurred.

### Known limits
Validation is the middle amplitude regime (STL finding), so tuned settings may not transfer to test. The last h
validation labels point into the test period (at most 6 of 1,728 rows; never trained on). The Random Forest baseline
moved up to 0.94% on val from 6 fewer training rows: gaps of that size are not evidence. Phase 3 MLflow runs used
the unpurged data and are slightly stale against the fixture.

## Phase 4: pre-run measurements, 2026-09-24

Measured at HEAD 6655b54 (LSTMForecaster committed; selection module not yet), before any tuning run. This entry records
measurements only. It does not modify the pre-registration (f31c3aa) or the selection rule.

### What was run
An inline, uncommitted script fitted LSTMForecaster three times on data/processed/train_t6.csv only (13,860 eligible rows
after dropping NaN targets; 16 feature columns; 13,843 windows at L=18), each with seed=42, huber_delta=20.0, max_epochs=3
and the default architecture and optimizer settings (hidden_size 64, num_layers 1, dropout 0, learning_rate 1e-3,
batch_size 64). Two fits used torch_num_threads=4 (A4, B4); one used torch_num_threads=1 (C1). Each fit was followed by
predict on the same train rows. No validation or test file was opened. Machine: Linux x86_64, 12 cores, 7,115 MB RAM
(torch default 8 threads).

### Repeatability at 4 threads
A4 and B4 produced bitwise-identical predictions (max absolute difference 0.0 Wh; NaN positions equal) and bitwise-identical
per-epoch training losses [847.262, 780.865, 749.013] (printed to 3 decimals).

### Thread count changes the result
C1 (1 thread) differed from A4 (4 threads): the epoch-1 loss agreed at the printed precision (847.262); the epoch-2 and
epoch-3 losses were 781.03 and 748.192 against 780.865 and 749.013; predictions differed by up to 35.36 Wh after 3 epochs.
The cause of this difference was not isolated, and no explanation is claimed. The protocol fixes torch_num_threads = 4 and
records it in params; that is unchanged.

### Cost
Wall time per epoch: A4 1.00 s, B4 0.65 s, C1 1.16 s (range 0.65-1.16 s/epoch; the two identical 4-thread fits differed
in time). Peak process RSS 554-569 MB during the three fits (344 MB before them), approximately 570 MB.

### Reference-loader honesty point
load_val_reference (src/tuning/selection.py, commit a7cfb2f) parses the whole golden fixture, which also contains the
linear_regression|h6 test metrics, so the parse loads them into memory. The function indexes only val_mae and val_rmse and
returns only those two values; a synthetic-fixture test (with distinctive test values) and a grep gate (no test metric names
in the module) check this. The isolation guarantee is about what is indexed and returned, not about what the file parse loads.

### Test evaluations to date
Zero. No LSTM has been fitted or evaluated on the real validation or test partitions, and no test metric or test
prediction has been computed for any LSTM. The measurement fits above used the train partition only; unit tests use
synthetic data only.

## Phase 4: runner test-isolation decision (Option 2), 2026-09-24

Decided at HEAD d6351d7, before any runner or validation-only code exists. Supplements the pre-registration (f31c3aa); it
does not modify it or the selection rule (a7cfb2f).

### Decision
The tuning runner is structurally validation-only: it receives train and validation data only, and has no code path that
opens, loads, evaluates, or persists test data, test metrics, or test predictions.

### Why
The pre-registration requires that tuning never read, persist, log, or use test metrics or predictions. evaluate_forecaster
computes train, val and test metrics on every call and needs all three partitions as input. Using it as-is (Option 1) would
comply, but only by discipline: the runner would load test labels and hold test predictions in its result object. A
structurally validation-only path lets a test demonstrate the rule instead of relying on it.

### Conditions
- The validation-only evaluation is a separate module (planned: src/evaluation/validation_only.py). harness.py is not modified.
- Its scoring is tested for equivalence against the existing harness: on stub forecasters including one with required_history_length
  greater than 0, and on the four Phase 3 baselines against the fixture's validation metrics and evaluated counts, under the
  same two-tier bar as the golden test (counts exact; metrics exact except linear_regression at relative 1e-12).
- The runner takes explicit train and validation file paths (no directory scan, no glob), so no test path is ever constructed.
- An isolation test runs the runner where only train and validation files exist and requires it to complete. Grep gates check
  that the runner and the validation-only module contain no test data path, no import of the harness, and no use of
  mlflow_logger.

### Accepted costs and limits
- About ten lines of prediction logic are duplicated from the harness; the equivalence tests are the guard against drift.
- This constrains the runner and its inputs. It does not prevent a different script from reading test data.
- The final evaluation (once per seed, per the pre-registration) uses the full harness, deliberately.

### Status
No validation-only module or runner exists. Zero LSTM test evaluations to date.

## Phase 4: final evaluation result (pre-registered protocol), 2026-09-24

Recorded at HEAD 0c685de. Produced by src/evaluation/final_evaluation.py (2cfaa55) under the protocol registered in f31c3aa,
from the configuration selected in results/tuning_h6_validation.json (c24fa69). Results: results/final_evaluation.json and
results/final_evaluation_predictions_h6.csv (0c685de). This entry states results and limits; it claims no cause.

### Tier 1 verdict (h=6, primary)
Verdict as computed by tier1_verdict with the registered <= comparison: NOT SHOWN.
Reference (linear_regression h=6, test, unrounded): MAE 42.39131825471337, RMSE 80.61873658942119. Thresholds (0.99 x): MAE 41.96740507216624, RMSE 79.81254922352697.

| seed | test MAE | MAE <= threshold | test RMSE | RMSE <= threshold |
|---|---|---|---|---|
| 42 | 39.313 | yes | 84.187 | no |
| 43 | 38.094 | yes | 81.017 | no |
| 44 | 38.788 | yes | 85.228 | no |

All three seeds must pass both metrics independently; a mean cannot rescue a failing seed. Either verdict is a registered,
reportable outcome.

### Test evaluations
Six (3 seeds x 2 horizons), no reruns, no parameter changed after any result. Any further use of the test partition needs a new
dated registration that states this count.

### Tier 2 (reporting only; mean and sample standard deviation, ddof=1)
LSTM, h=6 (primary):

| seed | test MAE | test RMSE | test MAPE |
|---|---|---|---|
| 42 | 39.313 | 84.187 | 33.017 |
| 43 | 38.094 | 81.017 | 31.344 |
| 44 | 38.788 | 85.228 | 31.639 |
| mean | 38.731 | 83.477 | 32.000 |
| std (ddof=1) | 0.612 | 2.193 | 0.893 |

LSTM, h=1 (secondary), same layout:

| seed | test MAE | test RMSE | test MAPE |
|---|---|---|---|
| 42 | 28.449 | 65.403 | 23.186 |
| 43 | 28.456 | 65.034 | 23.639 |
| 44 | 28.700 | 65.883 | 23.754 |
| mean | 28.535 | 65.440 | 23.526 |
| std (ddof=1) | 0.143 | 0.426 | 0.300 |

Phase 3 baselines on the test partition (fixture values, fixed order, not ranked):

| model | horizon | test MAE | test RMSE | test MAPE |
|---|---|---|---|---|
| naive_persistence | h=1 | 26.496 | 66.142 | 21.656 |
| naive_persistence | h=6 | 47.600 | 103.625 | 40.429 |
| naive_seasonal | h=1 | 53.719 | 112.289 | 51.132 |
| naive_seasonal | h=6 | 53.776 | 112.358 | 51.177 |
| linear_regression | h=1 | 27.155 | 59.712 | 24.904 |
| linear_regression | h=6 | 42.391 | 80.619 | 41.704 |
| random_forest | h=1 | 32.460 | 65.866 | 30.572 |
| random_forest | h=6 | 45.118 | 83.065 | 45.340 |

### Validation vs test, selected configuration (h=6, ratio to the linear_regression reference of the same split)

| split | LR reference MAE | LR reference RMSE | LSTM seed-mean MAE | LSTM seed-mean RMSE | MAE ratio | RMSE ratio |
|---|---|---|---|---|---|---|
| validation | 44.416 | 81.931 | 34.860 | 79.721 | 0.7849 | 0.9730 |
| test | 42.391 | 80.619 | 38.731 | 83.477 | 0.9137 | 1.0355 |

### Known limits
- The selected configuration is the best of 12 by seed-mean validation score, so its validation score is optimistically biased.
- The two-metric selection score was decided by the RMSE ratio in all 12 configurations.
- On validation, the RMSE seed spread of the selected configuration (std 2.40) was about as large as its margin over the
linear_regression validation reference (2.21).
- Validation is the middle amplitude regime and test the lowest (STL finding, Phase 1); the last h validation labels point into
the test period (never trained on).
- The validation-only evaluator's bit-for-bit linear-regression test does not detect a Fortran-ordered input array (measured at
100 and 1,728 rows; cause not investigated).
- final_evaluation.py was verified by 22 tests and mechanical checks (step order, single fit path, no ranking constructs); its
full text was not read by the reviewer.
- Phase 3 MLflow runs used the unpurged data and are slightly stale against the fixture.

### Not concluded
No cause is claimed for any difference between the validation and test results; a selection effect and a regime difference are
both consistent with the ratios above, and nothing in this record separates them. This entry registers no follow-up experiment.
Exploratory analysis of the saved predictions may follow, is labelled exploratory, and cannot change the verdict.
