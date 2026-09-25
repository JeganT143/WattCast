"""Entrypoint for the WattCast Streamlit project report: a multi-page app
wiring five pages (report_pages/) via st.Page/st.navigation (Streamlit
1.36+ API — confirmed available at the installed version, 1.64.0).

DEV-CONVENIENCE ONLY (applies to the Live Inference page): that page starts
the FastAPI server itself, as a subprocess, so a single
`streamlit run streamlit_app.py` gives a working demo. This is NOT how you
would deploy this for real users — a real deployment runs the FastAPI
service as its own long-lived process (e.g. behind uvicorn/gunicorn, its
own container) and points a separately-deployed UI at its URL. Bundling
them here only avoids needing two terminals for a local demo.

All timestamps/values used anywhere in this app are SYNTHETIC (2016-04-30
00:00 onward, on the 10-minute grid) — no test-partition row is ever read
or displayed.
"""

import streamlit as st

from report_pages import data_features, limitations, live_inference, model_comparison, overview
from report_pages._style import inject_base_styles

st.set_page_config(page_title="WattCast project report", layout="centered")
inject_base_styles()

with st.sidebar:
    st.markdown("### WattCast")
    st.caption("Appliance energy forecasting — project report")

pages = [
    st.Page(overview.render, title="Overview", url_path="overview", default=True),
    st.Page(data_features.render, title="Data & Features", url_path="data-features"),
    st.Page(model_comparison.render, title="Model Comparison", url_path="model-comparison"),
    st.Page(live_inference.render, title="Live Inference", url_path="live-inference"),
    st.Page(limitations.render, title="Limitations & Design", url_path="limitations-design"),
]

navigation = st.navigation(pages)
navigation.run()
