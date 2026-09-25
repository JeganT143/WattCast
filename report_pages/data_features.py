"""Data & Features page: the registered Phase 1 EDA findings and the
Phase 2 feature schema, both read directly from their source files via
src/ui/report_data.py — no claim here is drafted independently of a file.
"""

import streamlit as st

from src.ui.report_data import load_eda_summary, load_feature_config

_FEATURE_EXPLANATIONS = {
    "lags": "Past values of Appliances at fixed offsets back from the forecast origin "
    "(e.g. 1, 2, ... 6 steps, and 144 steps — one day earlier at 10-minute resolution).",
    "rolling": "Backward-only rolling mean/std over a trailing window of past "
    "Appliances values — never centered, since a live buffer has no future rows.",
    "cyclical": "Time-of-day and day-of-week position, encoded as sin/cos pairs "
    "(and as raw hour/day-of-week/is_weekend for tree models) — always knowable in "
    "advance, so these carry no leakage risk regardless of forecast horizon.",
}


def render() -> None:
    st.title("WattCast — Data & Features")

    st.subheader("EDA findings (Phase 1)")
    st.caption('Source: DECISIONS.md, "## Phase 1 — EDA Key Findings"')
    for bullet in load_eda_summary():
        st.markdown(f"- {bullet}")

    st.subheader("Feature schema (Phase 2)")
    st.caption("Source: config/features.py")
    config = load_feature_config()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Lag steps:** `{config['lag_steps']}`")
        st.markdown(_FEATURE_EXPLANATIONS["lags"])
    with col2:
        st.markdown(f"**Rolling windows:** `{config['rolling_windows']}`")
        st.markdown(_FEATURE_EXPLANATIONS["rolling"])

    st.markdown("**Cyclical / calendar features**")
    st.markdown(_FEATURE_EXPLANATIONS["cyclical"])

    with st.expander(f"All {len(config['feature_columns'])} feature columns"):
        st.code("\n".join(config["feature_columns"]))

    st.subheader("Leakage-safety discipline")
    st.markdown(
        "- **Train-only scaler**: `StandardScaler` is fit only on train-partition rows "
        "and applied unchanged to validation/test/serving rows "
        "(`src/preprocessing/scaling.py`) — val/test statistics never leak into the "
        "fitted scaler.\n"
        "- **Backward-only rolling windows**: rolling mean/std always use `center=False` "
        "(`src/features/rolling.py`) — a centered window is undefined at serving time, "
        "since a live context buffer has no future rows.\n"
        "- **Structural, not conventional**: lag/rolling code rejects non-positive "
        "shift/window arguments, and target code rejects non-positive horizons "
        "(`src/features/lag.py`, `src/features/targets.py`, `src/features/rolling.py`) "
        "— the leakage boundary is enforced in code, not just documented."
    )
