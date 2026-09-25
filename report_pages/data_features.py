"""Data & features: the dataset, the chronological split, the exploratory
findings, and the feature contract of the deployed models."""

import pandas as pd
import streamlit as st

from report_pages._style import page_header, source_note
from src.ui.content import DATASET, EDA_FINDINGS, LEAKAGE_SAFEGUARDS, PARTITIONS
from src.ui.report_data import load_bundle_schemas, load_feature_config, model_label


def _feature_table(feature_columns: list[str], schemas: dict[str, dict]) -> pd.DataFrame:
    def kind(column: str) -> str:
        if "_lag_" in column:
            return "Lag"
        if "_roll" in column:
            return "Rolling statistic"
        return "Calendar"

    def used_by(column: str) -> str:
        users = [model_label(f) for f, s in schemas.items() if column in s["required_columns"]]
        return "All five" if len(users) == len(schemas) else ", ".join(users)

    return pd.DataFrame(
        {"Feature": feature_columns, "Type": [kind(c) for c in feature_columns], "Used by": [used_by(c) for c in feature_columns]}
    )


def render() -> None:
    page_header("Data & features", "The dataset, what exploration showed, and the inputs the models see.")

    st.subheader("Dataset", anchor=False)
    c1, c2, c3 = st.columns(3)
    c1.metric("Readings", f"{DATASET['n_rows']:,}")
    c2.metric("Interval", f"{DATASET['interval_minutes']} min")
    c3.metric("Period", f"{DATASET['span_months']} months")
    st.markdown(
        f"The [{DATASET['name']}]({DATASET['url']}) dataset ({DATASET['authors']}, CC BY 4.0) records one "
        f"house from {DATASET['start']} to {DATASET['end']}. The target is `{DATASET['target']}`, the energy "
        f"used by appliances in each 10-minute interval ({DATASET['target_unit']}). The other "
        f"{DATASET['n_columns'] - 2} columns (room temperatures and humidity, weather, lights, two random-noise "
        "control columns) were explored but are not model inputs."
    )

    st.markdown("**Chronological split.** No shuffling: every partition lies strictly after the one before it.")
    st.dataframe(pd.DataFrame(PARTITIONS), hide_index=True, width="stretch")

    st.subheader("What exploration showed", anchor=False)
    st.markdown("\n".join(f"- {item}" for item in EDA_FINDINGS))
    source_note("notebooks/01_eda.ipynb")

    st.subheader("Features", anchor=False)
    config = load_feature_config()
    lags = ", ".join(str(n) for n in config["lag_steps"])
    windows = " and ".join(str(n) for n in config["rolling_windows"])
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True, height="stretch"):
        st.markdown(f"**Lags**  \nReadings {lags} steps back. Step 144 is the same time yesterday.")
    with c2.container(border=True, height="stretch"):
        st.markdown(f"**Rolling statistics**  \nMean and standard deviation over the last {windows} readings (1 h and 3 h).")
    with c3.container(border=True, height="stretch"):
        st.markdown("**Calendar**  \nHour, weekday, weekend flag, and sine/cosine encodings of time of day and weekday.")

    schemas = load_bundle_schemas()
    with st.expander(f"All {len(config['feature_columns'])} features and which models use them"):
        st.dataframe(_feature_table(config["feature_columns"], schemas), hide_index=True, width="stretch")
        st.caption(
            "Sequence models (LSTM, GRU, CNN-LSTM) read windows of 18 consecutive feature rows and drop the raw "
            "hour and weekday integers, keeping their sine/cosine encodings. Lag and rolling features are "
            "standardised with a scaler fit on training rows only."
        )
    source_note("config/features.py and models/*/schema.json")

    st.subheader("Leakage safeguards", anchor=False)
    st.markdown(
        "Every feature must be known at the moment the forecast is made. These rules are enforced in code, "
        "not left to convention:"
    )
    st.markdown("\n".join(f"- {item}" for item in LEAKAGE_SAFEGUARDS))
