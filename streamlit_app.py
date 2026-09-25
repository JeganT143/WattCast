"""WattCast Streamlit app — entry point.

    streamlit run streamlit_app.py

Five pages (report_pages/) behind top navigation. The Live forecast page
runs the serving core in-process from the committed bundles in models/, so
the app needs no API server, MLflow store, or raw dataset.
"""

import streamlit as st

from report_pages import data_features, limitations, live_inference, model_comparison, overview
from report_pages._style import inject_base_styles

st.set_page_config(
    page_title="WattCast — appliance energy forecasting",
    page_icon=":material/bolt:",
    layout="centered",
)
inject_base_styles()

pages = [
    st.Page(overview.render, title="Overview", icon=":material/home:", default=True),
    st.Page(live_inference.render, title="Live forecast", icon=":material/show_chart:", url_path="live-forecast"),
    st.Page(model_comparison.render, title="Model comparison", icon=":material/leaderboard:", url_path="model-comparison"),
    st.Page(data_features.render, title="Data & features", icon=":material/dataset:", url_path="data-features"),
    st.Page(limitations.render, title="Method & limitations", icon=":material/fact_check:", url_path="method"),
]

st.navigation(pages, position="top").run()
