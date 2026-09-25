"""Overview page: project thesis, dataset summary, and an architecture
diagram. The diagram uses an HTML/CSS box-and-arrow layout via
st.components.v1.html rather than st.graphviz_chart, because the graphviz
system binary (`dot`) is not installed in this environment (confirmed via
`shutil.which("dot")` returning None) — st.graphviz_chart requires it to
render, so a self-contained HTML fallback was used instead.
"""

import streamlit as st
import streamlit.components.v1 as components

from src.ui.report_data import load_dataset_summary

_ARCHITECTURE_HTML = """
<div style="font-family: sans-serif; display: flex; flex-direction: column; gap: 8px; padding: 8px;">
  <style>
    .wc-row { display: flex; align-items: center; justify-content: center; gap: 10px; flex-wrap: wrap; }
    .wc-box {
      border: 1px solid #888; border-radius: 6px; padding: 10px 14px;
      background: #f5f5f5; color: #111; font-size: 13px; text-align: center;
      min-width: 120px;
    }
    .wc-arrow { font-size: 20px; color: #666; }
    @media (prefers-color-scheme: dark) {
      .wc-box { background: #2b2b2b; color: #eee; border-color: #666; }
      .wc-arrow { color: #aaa; }
    }
  </style>
  <div class="wc-row">
    <div class="wc-box">Raw CSV<br/>(19,735 rows, 10-min)</div>
    <div class="wc-arrow">&rarr;</div>
    <div class="wc-box">Feature engineering<br/>lags / rolling stats / cyclical</div>
    <div class="wc-arrow">&rarr;</div>
    <div class="wc-box">Train-only<br/>StandardScaler</div>
  </div>
  <div class="wc-row">
    <div class="wc-arrow" style="transform: rotate(90deg);">&rarr;</div>
  </div>
  <div class="wc-row">
    <div class="wc-box">Forecaster models<br/>LR / RF / LSTM / GRU / CNN-LSTM</div>
    <div class="wc-arrow">&rarr;</div>
    <div class="wc-box">MLflow registry<br/>(@champion alias, 5 families)</div>
  </div>
  <div class="wc-row">
    <div class="wc-arrow" style="transform: rotate(90deg);">&rarr;</div>
  </div>
  <div class="wc-row">
    <div class="wc-box">FastAPI serving<br/>/ingest, /predict, /health, /model</div>
    <div class="wc-arrow">&rarr;</div>
    <div class="wc-box">This Streamlit report<br/>(dev-convenience subprocess)</div>
  </div>
</div>
"""


def render() -> None:
    st.title("WattCast — Overview")

    st.markdown(
        "WattCast forecasts household appliance energy consumption. The project "
        "deliberately starts with naive and linear baselines before any deep model, "
        "so every later result is judged against a documented, reproducible reference "
        "rather than against nothing. Every feature respects a strict "
        "forecast-origin rule: nothing after the prediction time may enter a feature, "
        "enforced structurally in the feature-engineering code, not just by convention. "
        "The same `Forecaster` interface is used for training, evaluation, and now live "
        "serving, so the models compared here are exactly what is registered and served."
    )

    st.subheader("Dataset")
    summary = load_dataset_summary()
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{summary['n_rows']:,}")
    col2.metric("Columns", summary["n_columns"])
    col3.metric("Interval", f"{summary['interval_minutes']} min")
    st.caption(
        f"UCI Appliances Energy Prediction dataset, {summary['start_date']} to "
        f"{summary['end_date']} (~{summary['span_months']} months). Source: {summary['source']}."
    )

    st.subheader("Architecture")
    st.caption(
        "Rendered as HTML/CSS (not Graphviz): the `dot` binary is not installed in "
        "this environment (`shutil.which(\"dot\")` returned None)."
    )
    components.html(_ARCHITECTURE_HTML, height=340, scrolling=False)
