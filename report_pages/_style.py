"""Shared visual styling for the report: a small base stylesheet injected
once from the entrypoint (streamlit_app.py), plus a page_header() helper so
every page gets the same title/subtitle/divider treatment. Purely
presentational — no page content or data logic lives here.
"""

import streamlit as st

_BASE_CSS = """
<style>
h1, h2, h3 { font-weight: 600; }

/* This app pins a single light theme (.streamlit/config.toml, base="light"),
   so every custom color below is an explicit background+text pair designed
   for that theme — never a bare background with an inherited text color,
   and no prefers-color-scheme override. Mixing an OS/browser dark
   preference with a pinned-light Streamlit theme is exactly what caused
   low-contrast (dark-on-dark / light-on-light) text before this fix. */

.wc-subtitle {
    color: #4A5568;
    font-size: 0.95rem;
    margin-top: -0.4rem;
    margin-bottom: 0.6rem;
}

.wc-callout {
    background: #F4F6F8;
    color: #1A1A1A;
    border-left: 4px solid #1F4E79;
    border-radius: 6px;
    padding: 0.9rem 1.1rem;
    margin: 0.6rem 0 1rem 0;
    font-size: 0.94rem;
    line-height: 1.5;
}

.wc-source {
    color: #6B7280;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    margin-bottom: 0.3rem;
}

div[data-testid="stMetric"] {
    background: #F4F6F8;
    border: 1px solid #E3E7EB;
    border-radius: 8px;
    padding: 0.8rem 1rem 0.5rem 1rem;
}
</style>
"""


def inject_base_styles() -> None:
    st.markdown(_BASE_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f'<div class="wc-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    st.divider()


def source_caption(text: str) -> None:
    st.markdown(f'<div class="wc-source">Source · {text}</div>', unsafe_allow_html=True)


def callout(markdown_text: str) -> None:
    st.markdown(f'<div class="wc-callout">{markdown_text}</div>', unsafe_allow_html=True)
