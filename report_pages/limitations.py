"""Method & limitations: the evaluation protocol, how serving works, the
deployed model versions, and what the results do not show."""

import pandas as pd
import streamlit as st

from report_pages._style import page_header
from src.ui.content import LIMITATIONS, REPO_URL
from src.ui.report_data import load_bundle_schemas, model_label

_PROTOCOL = [
    "**Rules before results.** Success criteria, seeds, hyperparameter grids and comparison rules were written "
    "down and committed before the experiments they govern were run.",
    "**Two metrics, every seed.** A model counts as better only if it improves both MAE and RMSE by at least 1% "
    "for every training seed. A better mean cannot rescue a failing seed.",
    "**Test data used once.** The LSTM's test-set check ran once per seed and was never repeated or re-tuned. "
    "Model selection used the validation period or walk-forward folds only.",
    "**Tuning cannot see test data.** Hyperparameter search runs through a separate validation-only code path "
    "that is never given the test partition.",
]

_ENDPOINTS = pd.DataFrame(
    [
        {"Endpoint": "POST /ingest", "Purpose": "Append the next 10-minute reading (must be exactly 10 min after the last)"},
        {"Endpoint": "POST /predict", "Purpose": "Forecast 60 min ahead for the requested model families (read-only)"},
        {"Endpoint": "GET /health", "Purpose": "Readiness, buffer size and the time of the latest reading"},
        {"Endpoint": "GET /model", "Purpose": "Primary model schema and the list of available families"},
    ]
)


def _served_models() -> pd.DataFrame:
    rows = []
    for family, schema in load_bundle_schemas().items():
        rows.append(
            {
                "Model": model_label(family),
                "Version": f"v{schema.get('model_version', '?')}",
                "Trained on": f"{schema['train_start'][:10]} → {schema['train_end'][:10]}",
                "Training rows": f"{schema['n_fit_rows']:,}",
                "Inputs": len(schema["required_columns"]),
                "History needed": f"{schema['required_raw_history']} readings",
            }
        )
    return pd.DataFrame(rows)


def render() -> None:
    page_header("Method & limitations", "How the results were produced, how serving works, and what they do not show.")

    st.subheader("Evaluation protocol", anchor=False)
    st.markdown("\n".join(f"- {item}" for item in _PROTOCOL))

    st.subheader("Serving", anchor=False)
    st.markdown(
        "Every model implements one `Forecaster` interface, so training, evaluation and serving share the same "
        "feature code. At serving time a rolling buffer holds the latest 162 readings (27 hours), enough for a "
        "one-day lag plus the sequence models' 18-row window. Each new reading must arrive exactly 10 minutes "
        "after the previous one. The same serving core backs this app and a FastAPI service with four endpoints:"
    )
    st.dataframe(_ENDPOINTS, hide_index=True, width="stretch")

    st.markdown("**Deployed models.** All five were trained on the training and validation periods combined; "
                "the test period was never used for fitting.")
    st.dataframe(_served_models(), hide_index=True, width="stretch")

    st.subheader("Limitations", anchor=False)
    st.markdown("\n".join(f"- {item}" for item in LIMITATIONS))

    st.subheader("Learn more", anchor=False)
    st.markdown(
        f"- [Source code and setup]({REPO_URL})\n"
        f"- [Project report]({REPO_URL}/blob/master/report.md): full write-up of method and results\n"
        f"- [Engineering decisions]({REPO_URL}/blob/master/decisions.md): why each major choice was made"
    )
