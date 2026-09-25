"""Overview page: project thesis, dataset summary, and an architecture
diagram. The diagram uses an HTML/CSS box-and-arrow layout via
st.components.v1.html rather than st.graphviz_chart, because the graphviz
system binary (`dot`) is not installed in this environment (confirmed via
`shutil.which("dot")` returning None) — st.graphviz_chart requires it to
render, so a self-contained HTML fallback was used instead.
"""

import streamlit as st
import streamlit.components.v1 as components

from report_pages._style import callout, page_header, source_caption
from src.ui.report_data import load_dataset_summary

_ARCHITECTURE_HTML = """
<div class="wc-pipeline">
  <style>
    html, body { background: #FFFFFF; margin: 0; }
    .wc-pipeline {
      display: flex; flex-direction: column; align-items: center;
      font-family: -apple-system, "Segoe UI", sans-serif; padding: 4px 0 8px 0;
      background: #FFFFFF;
    }
    .wc-step {
      width: 92%; max-width: 460px; background: #F4F6F8;
      border: 1px solid #E3E7EB; border-left: 4px solid #1F4E79;
      border-radius: 8px; padding: 10px 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .wc-step h4 { margin: 0 0 3px 0; font-size: 13.5px; color: #1F4E79; font-weight: 600; }
    .wc-step p { margin: 0; font-size: 12.5px; color: #3A3F45; line-height: 1.4; }
    .wc-connector {
      width: 2px; height: 20px; background: #C7CDD3; margin: 2px 0;
    }
  </style>
  <div class="wc-step"><h4>1 · Raw data</h4><p>19,735 rows, 10-minute cadence — UCI Appliances Energy Prediction</p></div>
  <div class="wc-connector"></div>
  <div class="wc-step"><h4>2 · Feature engineering</h4><p>Lags, backward-only rolling stats, cyclical calendar encodings</p></div>
  <div class="wc-connector"></div>
  <div class="wc-step"><h4>3 · Train-only scaling</h4><p>StandardScaler fit on the train partition only, applied unchanged elsewhere</p></div>
  <div class="wc-connector"></div>
  <div class="wc-step"><h4>4 · Forecaster models</h4><p>Linear Regression, Random Forest, LSTM, GRU, CNN-LSTM — one shared interface</p></div>
  <div class="wc-connector"></div>
  <div class="wc-step"><h4>5 · MLflow registry</h4><p>Each family registered independently under its own @champion alias</p></div>
  <div class="wc-connector"></div>
  <div class="wc-step"><h4>6 · FastAPI serving</h4><p>/ingest &middot; /predict &middot; /health &middot; /model</p></div>
  <div class="wc-connector"></div>
  <div class="wc-step"><h4>7 · This report</h4><p>Live Inference page calls the API above through a thin HTTP client</p></div>
</div>
"""


def render() -> None:
    page_header("Overview", "What WattCast is, what it forecasts, and how the pieces fit together")

    callout(
        "WattCast forecasts household appliance energy consumption. The project "
        "deliberately starts with naive and linear baselines before any deep model, "
        "so every later result is judged against a documented, reproducible reference "
        "rather than against nothing. Every feature respects a strict forecast-origin "
        "rule &mdash; nothing after the prediction time may enter a feature &mdash; "
        "enforced structurally in the feature-engineering code, not just by convention. "
        "The same <code>Forecaster</code> interface is used for training, evaluation, "
        "and live serving, so the models compared in this report are exactly what is "
        "registered and served."
    )

    st.markdown("#### Dataset")
    summary = load_dataset_summary()
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{summary['n_rows']:,}")
    col2.metric("Columns", summary["n_columns"])
    col3.metric("Sampling interval", f"{summary['interval_minutes']} min")
    source_caption(
        f"{summary['source']} · UCI Appliances Energy Prediction dataset, "
        f"{summary['start_date']} to {summary['end_date']} (~{summary['span_months']} months)"
    )

    st.markdown("#### Architecture")
    source_caption(
        'HTML/CSS diagram (graphviz\'s "dot" binary is not installed in this environment)'
    )
    components.html(_ARCHITECTURE_HTML, height=640, scrolling=False)
