"""Shared presentation helpers: base stylesheet, page header, callouts and
the per-model colour palette. No data loading or page content lives here."""

import streamlit as st

# One colour per model family, used by every chart in the app. The deep
# models match the colours of the walk-forward figures in docs/img/.
MODEL_COLORS = {
    "naive_persistence": "#B8BEC6",
    "naive_seasonal": "#8A929C",
    "linear_regression": "#1F4E79",
    "random_forest": "#A6761D",
    "lstm": "#1B9E77",
    "gru": "#D95F02",
    "cnn_lstm": "#7570B3",
}

GROUP_COLORS = {"Naive baseline": "#B8BEC6", "Classical": "#1F4E79", "Deep sequence": "#1B9E77"}

# The theme is pinned to light (.streamlit/config.toml), so every custom
# colour below is an explicit background + text pair designed for it.
_BASE_CSS = """
<style>
[data-testid="stMainBlockContainer"] { max-width: 58rem; padding-top: 4.5rem; }
h1, h2, h3, h4 { font-weight: 650; letter-spacing: -0.01em; }
.wc-kicker {
    color: #1F4E79; font-size: 0.78rem; font-weight: 650;
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.1rem;
}
.wc-subtitle { color: #4A5568; font-size: 1.05rem; line-height: 1.5; margin: -0.3rem 0 1.4rem 0; }
.wc-callout {
    background: #F1F5F9; color: #1A202C; border-left: 4px solid #1F4E79;
    border-radius: 6px; padding: 0.85rem 1.1rem; margin: 0.4rem 0 1.1rem 0;
    font-size: 0.95rem; line-height: 1.55;
}
.wc-callout.wc-warn { background: #FFF8EB; border-left-color: #B7791F; }
.wc-source { color: #6B7280; font-size: 0.8rem; margin: -0.4rem 0 0.8rem 0; }
div[data-testid="stMetric"] {
    background: #F7F9FB; border: 1px solid #E3E7EB; border-radius: 8px;
    padding: 0.75rem 1rem 0.6rem 1rem;
}
div[data-testid="stMetricValue"] { font-size: 1.65rem; }
</style>
"""


def inject_base_styles() -> None:
    st.markdown(_BASE_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str, kicker: str | None = None) -> None:
    if kicker:
        st.markdown(f'<div class="wc-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title, anchor=False)
    st.markdown(f'<div class="wc-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def callout(html: str, warn: bool = False) -> None:
    css_class = "wc-callout wc-warn" if warn else "wc-callout"
    st.markdown(f'<div class="{css_class}">{html}</div>', unsafe_allow_html=True)


def source_note(text: str) -> None:
    st.markdown(f'<div class="wc-source">Source: {text}</div>', unsafe_allow_html=True)
