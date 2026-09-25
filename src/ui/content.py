"""Static narrative content shown by the Streamlit app.

Numbers here restate recorded findings (notebooks/01_eda.ipynb and the
decision record in decisions.md); anything derivable from the raw data is
cross-checked against it in tests/test_ui_content.py. Result metrics are
never written here — they are read from results/ by src/ui/report_data.py.
"""

REPO_URL = "https://github.com/JeganT143/WattCast"

DATASET = {
    "name": "UCI Appliances Energy Prediction",
    "url": "https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction",
    "authors": "Candanedo et al.",
    "n_rows": 19_735,
    "n_columns": 29,
    "interval_minutes": 10,
    "start": "2016-01-11",
    "end": "2016-05-27",
    "span_months": 4.5,
    "target": "Appliances",
    "target_unit": "Wh",
}

# Chronological partitions (config/features.py SPLIT_TRAIN_END / SPLIT_VAL_END).
PARTITIONS = [
    {"Partition": "Train", "Period": "2016-01-11 → 2016-04-17", "Used for": "Fitting models and the scaler"},
    {"Partition": "Validation", "Period": "2016-04-18 → 2016-04-29", "Used for": "LSTM hyperparameter selection"},
    {"Partition": "Test", "Period": "2016-04-30 → 2016-05-27", "Used for": "One pre-registered check, once per seed"},
]

EDA_FINDINGS = [
    "**Clean timeline.** Strict 10-minute cadence with no missing values and no duplicate or missing timestamps, "
    "so lag and rolling features need no gap handling.",
    "**Strong daily cycle.** Overnight trough, morning rise and an evening peak around 18:00. Weekday and weekend "
    "profiles differ in shape, not just level.",
    "**Short memory plus a daily echo.** Autocorrelation fades within hours but recurs at the daily lag "
    "(144 steps), so a one-day lag is kept as a candidate feature.",
    "**Extremes are real.** Readings up to 1,080 Wh sit inside plausible ramps; there is no evidence of sensor "
    "error, so the target is kept unclipped.",
    "**Drifting amplitude.** The daily rhythm keeps its shape over the 4.5 months, but its size shrinks "
    "(median daily P90–P10 range 94 → 74 → 60 Wh across early, middle and late thirds).",
    "**Stationary around a moving level.** The ADF test rejects a unit root (statistic −21.62), so no differencing "
    "is applied.",
    "**Signal is temporal.** Same-timestamp sensor readings correlate weakly with consumption (every sensor "
    "|r| < 0.25); recent history and time of day carry the predictive structure.",
]

LEAKAGE_SAFEGUARDS = [
    "**Features look backward only.** Lag and rolling helpers reject non-positive offsets, and rolling windows "
    "are never centered (`src/features/lag.py`, `src/features/rolling.py`).",
    "**Targets look forward only.** Target construction rejects non-positive horizons (`src/features/targets.py`).",
    "**Features before splitting.** Features are computed on the full continuous series, then split "
    "chronologically, so the first validation rows still see the tail of training history.",
    "**Train-only scaling.** The `StandardScaler` is fit on training rows only and applied unchanged everywhere "
    "else, including serving (`src/preprocessing/scaling.py`).",
    "**Label purge.** Training labels whose target time falls in a later partition are masked, never used "
    "(`src/data/purge.py`).",
]

LIMITATIONS = [
    "**One house, one period.** All results come from a single household over 138 calendar days. The walk-forward folds "
    "are consecutive slices of that period, not independent datasets.",
    "**One-sided tuning.** Only the LSTM had a hyperparameter search; GRU and CNN-LSTM reused its configuration "
    "untuned.",
    "**Validation overlap.** Walk-forward folds 7 and 8 overlap the validation period used to choose the LSTM "
    "configuration. A folds 1–6 sensitivity check gives the same classification.",
    "**Shifting regime.** Daily amplitude declines over the study period, so later folds and the test period "
    "behave differently from early training data.",
    "**Different loss functions.** Deep models train on Huber loss (δ = 40 Wh), linear regression on squared "
    "error. This may explain part of the MAE/RMSE split, but that was not tested.",
    "**Post-hoc champion.** Linear regression was chosen as the primary serving model after seeing walk-forward "
    "RMSE; its edge over random forest depends on a single fold.",
    "**Consumption history only.** Models use past `Appliances` values and calendar features; temperature, "
    "humidity and weather columns are not used.",
    "**Demo is not an evaluation.** The live page forecasts from readings you type in. Its forecasts are never "
    "compared with real outcomes.",
]
