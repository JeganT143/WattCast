"""Streamlit UI for comparing all five registered model families
(linear_regression, random_forest, lstm, gru, cnn_lstm) side by side against
the live FastAPI serving app (src/serving/app.py — unchanged by this file).

DEV-CONVENIENCE ONLY: this script starts the FastAPI server itself, as a
subprocess, so a single `streamlit run streamlit_app.py` gives a working
demo. This is NOT how you would deploy this for real users — a real
deployment runs the FastAPI service as its own long-lived process (e.g.
behind uvicorn/gunicorn, its own container) and points a separately-deployed
UI at its URL. Bundling them here only avoids needing two terminals for a
local demo.

All timestamps/values in this UI are SYNTHETIC (2016-04-30 00:00 onward, on
the 10-minute grid) — no test-partition row is ever read or displayed.
"""

import atexit
import time

import pandas as pd
import streamlit as st

from src.ui.client import (
    ApiClient,
    ApiError,
    compute_next_timestamp,
    find_free_port,
    start_server_subprocess,
    wait_for_health,
)

ALL_FAMILIES = ["linear_regression", "random_forest", "lstm", "gru", "cnn_lstm"]

CAVEAT_BANNER = (
    "**No model family has been shown to decisively outperform the others.** "
    "The pre-registered 8-fold walk-forward comparison (DECISIONS.md, \"Phase 5: "
    "walk-forward run results\") required a deep model (LSTM/GRU/CNN-LSTM) to beat "
    "each reference (linear_regression, random_forest) on **both** MAE and RMSE "
    "across all seeds to count as \"shown better\" — all six comparisons came back "
    "**\"not shown\"** (deep models were consistently better on MAE but worse on "
    "RMSE than the linear-regression reference). These predictions are shown here "
    "for side-by-side comparison only, not to declare a winner."
)


@st.cache_resource
def _start_server() -> str:
    """Starts the FastAPI serving subprocess exactly once per Streamlit
    server process (not once per script rerun, which is what a plain
    module-level call would do — st.cache_resource makes the underlying
    Popen a true singleton across reruns) and registers cleanup so it
    doesn't outlive this process. Deliberately does NOT wait for health here
    — that must be re-checked on every rerun (see main()), not cached,
    otherwise an early "not ready yet" result would be cached forever."""
    port = find_free_port()
    proc = start_server_subprocess(port)

    def _cleanup():
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    atexit.register(_cleanup)

    return f"http://127.0.0.1:{port}"


def main() -> None:
    st.set_page_config(page_title="WattCast: five-family comparison", layout="centered")
    st.title("WattCast — compare all five model families")
    st.markdown(CAVEAT_BANNER)

    base_url = _start_server()
    ready = wait_for_health(base_url, timeout=2.0)

    if not ready:
        st.warning("Starting server... this page will refresh automatically.")
        time.sleep(1.0)
        st.rerun()
        return

    client = ApiClient(base_url)
    health = client.health()
    next_ts = compute_next_timestamp(health)

    st.subheader("1. Ingest a synthetic observation")
    col1, col2 = st.columns(2)
    with col1:
        appliances_value = st.number_input(
            "Next Appliances value (Wh)", value=60.0, step=1.0
        )
    with col2:
        st.text_input(
            "Timestamp to ingest at (server-enforced, not editable)",
            value=str(next_ts),
            disabled=True,
        )

    if st.button("Ingest"):
        try:
            result = client.ingest(next_ts, appliances_value)
            st.success(
                f"Ingested at {result['origin_timestamp']} "
                f"(buffer {result['have']}/{result['need']})"
            )
        except ApiError as e:
            message = str(e)
            if "409" in message and "expected_next" in message:
                st.error(
                    "That timestamp was already ingested or is out of order. "
                    "The server's error detail: " + message
                )
            else:
                st.error(f"Ingest failed: {message}")

    st.subheader("2. Predict")
    selected_families = st.multiselect(
        "Model families", options=ALL_FAMILIES, default=ALL_FAMILIES
    )

    if st.button("Predict") and selected_families:
        try:
            response = client.predict(selected_families)
        except ApiError as e:
            message = str(e)
            if "503" in message:
                st.warning(
                    "Server reports insufficient history to predict yet "
                    "(should not happen given startup seeding): " + message
                )
            else:
                st.error(f"Predict failed: {message}")
        else:
            results = response["results"]
            ok_results = [r for r in results if "error" not in r]
            error_results = [r for r in results if "error" in r]

            if ok_results:
                table = pd.DataFrame(
                    [
                        {
                            "model_family": r["model_family"],
                            "prediction_wh": r["prediction_wh"],
                            "forecast_timestamp": r["forecast_timestamp"],
                        }
                        for r in ok_results
                    ]
                )
                st.dataframe(table, hide_index=True)
                st.bar_chart(table.set_index("model_family")["prediction_wh"])

            for r in error_results:
                st.warning(f"{r['model_family']}: not available ({r['error']})")


if __name__ == "__main__":
    main()
