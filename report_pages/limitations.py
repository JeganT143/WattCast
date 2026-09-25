"""Limitations & Design page: the registered limitations (DECISIONS.md,
"### Known limits (registered, restated)") and a brief note on the
Forecaster/Strategy pattern and the five served model families.
"""

import streamlit as st

from report_pages._style import page_header, source_caption
from src.ui.report_data import load_limitations


def render() -> None:
    page_header("Limitations & Design", "What this project does not claim, and how it is built")

    st.markdown("#### Registered limitations")
    source_caption('DECISIONS.md · "Known limits (registered, restated)" (Phase 5 walk-forward run-results)')
    with st.container(border=True):
        for item in load_limitations():
            st.markdown(f"- {item}")

    st.markdown("#### Design: the Forecaster / Strategy pattern")
    with st.container(border=True):
        st.markdown(
            "Every model in this project &mdash; naive baselines, linear regression, "
            "random forest, and the three sequence models (LSTM, GRU, CNN-LSTM) "
            "&mdash; implements the same `Forecaster` interface: `fit(X, y)`, "
            "`predict(X)`, `required_columns`, and `required_history_length`. The "
            "evaluation harness and the live serving code both depend only on this "
            "interface, never on a concrete model class, so adding a new family means "
            "writing a new `Forecaster` subclass, not touching the harness or the "
            "serving app."
        )
    with st.container(border=True):
        st.markdown(
            "Five families are currently registered and served live: "
            "`linear_regression`, `random_forest`, `lstm`, `gru`, and `cnn_lstm` "
            "&mdash; each independently trained, registered under its own MLflow "
            "`@champion` alias, and reachable through the same `/predict` endpoint "
            "on the Live Inference page."
        )
