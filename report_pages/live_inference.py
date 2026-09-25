"""Live forecast: the serving core (src/serving/service.py) running
in-process against the committed bundles in models/ (decisions.md, ADR-017).

The five models are loaded once per server process; every browser session
gets its own ServingService with a freshly seeded buffer, so visitors never
see or disturb each other's readings.
"""

import threading

import altair as alt
import pandas as pd
import streamlit as st

from config.paths import MODELS_DIR, SEED_HISTORY_PATH
from report_pages._style import MODEL_COLORS, callout, page_header
from src.serving.buffer import DuplicateTimestampError, NonSuccessorTimestampError
from src.serving.bundle import load_bundles
from src.serving.service import ServingService, create_service
from src.ui.report_data import model_label

STEP = pd.Timedelta(minutes=10)
HISTORY_POINTS = 36  # 6 hours of context in the chart
TIME_FORMAT = "%a %d %b %Y, %H:%M"


@st.cache_resource(show_spinner="Loading the five models…")
def _bundles() -> dict:
    return load_bundles(MODELS_DIR)


@st.cache_data
def _seed_history() -> pd.DataFrame:
    return pd.read_csv(SEED_HISTORY_PATH, parse_dates=["date"])


@st.cache_resource
def _predict_lock() -> threading.Lock:
    # Sequence models set torch's process-global thread count while predicting,
    # so predictions from concurrent sessions are serialized.
    return threading.Lock()


def _new_session() -> None:
    st.session_state.service = create_service(_bundles(), _seed_history())
    st.session_state.n_added = 0
    st.session_state.flash = None


def _service() -> ServingService:
    if "service" not in st.session_state:
        _new_session()
    return st.session_state.service


def _add_reading() -> None:
    service = _service()
    ts = service.buffer.last_timestamp + STEP
    value = float(st.session_state.reading_wh)
    try:
        service.ingest(ts, value)
    except (DuplicateTimestampError, NonSuccessorTimestampError, ValueError) as exc:
        st.session_state.flash = f"That reading was not accepted: {exc}"
        return
    st.session_state.n_added += 1
    st.session_state.flash = None
    st.toast(f"Added {value:g} Wh at {ts:%H:%M}", icon=":material/check:")


def _chart(frame: pd.DataFrame, results: list[dict], seed_end: pd.Timestamp) -> alt.LayerChart:
    history = frame.tail(HISTORY_POINTS).copy()
    added = history[history["date"] > seed_end]
    origin = history.iloc[-1]

    ok = [r for r in results if "error" not in r]
    end = max([origin["date"], *(r["forecast_timestamp"] for r in ok)]) + 2 * STEP
    domain = [f"{history['date'].iloc[0]:%Y-%m-%dT%H:%M:%S}", f"{end:%Y-%m-%dT%H:%M:%S}"]
    x = alt.X(
        "date:T",
        title=None,
        scale=alt.Scale(domain=domain),
        axis=alt.Axis(format="%H:%M", labelAngle=0, tickCount=8, grid=False),
    )
    y = alt.Y("Appliances:Q", title="Appliances energy (Wh)", scale=alt.Scale(zero=True))
    time_tip = alt.Tooltip("date:T", title="Time", format="%d %b %H:%M")

    layers = [
        alt.Chart(history).mark_line(color="#94A3B8", strokeWidth=1.8).encode(
            x=x, y=y, tooltip=[time_tip, alt.Tooltip("Appliances:Q", title="Reading (Wh)", format=".0f")]
        ),
        alt.Chart(pd.DataFrame({"date": [origin["date"]]})).mark_rule(color="#64748B", strokeDash=[4, 4]).encode(x=x),
        alt.Chart(pd.DataFrame({"date": [origin["date"]], "label": ["latest reading"]}))
        .mark_text(align="right", baseline="top", dx=-4, dy=4, color="#64748B", fontSize=11)
        .encode(x=x, y=alt.value(0), text="label:N"),
    ]
    if not added.empty:
        layers.append(
            alt.Chart(added).mark_circle(size=60, color="#1A202C").encode(
                x=x, y=y, tooltip=[time_tip, alt.Tooltip("Appliances:Q", title="Your reading (Wh)", format=".0f")]
            )
        )

    if ok:
        labels = [model_label(r["model_family"]) for r in ok]
        color = alt.Color(
            "Model:N",
            scale=alt.Scale(domain=labels, range=[MODEL_COLORS[r["model_family"]] for r in ok]),
            legend=alt.Legend(orient="bottom", title=None),
        )
        forecasts = pd.DataFrame(
            {"Model": labels, "date": [r["forecast_timestamp"] for r in ok], "Appliances": [r["prediction_wh"] for r in ok]}
        )
        links = pd.concat(
            [
                forecasts,
                forecasts.assign(date=origin["date"], Appliances=origin["Appliances"]),
            ]
        )
        layers += [
            alt.Chart(links).mark_line(strokeDash=[3, 3], strokeWidth=1.3, opacity=0.8).encode(
                x=x, y=y, color=color, detail="Model:N"
            ),
            alt.Chart(forecasts).mark_point(filled=True, size=110, shape="diamond", opacity=1).encode(
                x=x,
                y=y,
                color=color,
                tooltip=[
                    "Model:N",
                    alt.Tooltip("date:T", title="Forecast for", format="%d %b %H:%M"),
                    alt.Tooltip("Appliances:Q", title="Forecast (Wh)", format=".1f"),
                ],
            ),
        ]
    return alt.layer(*layers).properties(height=360)


def render() -> None:
    page_header(
        "Live forecast",
        "Play the meter: add the next 10-minute reading and compare every model's forecast for one hour later.",
    )

    try:
        service = _service()
    except Exception as exc:  # missing or unreadable model files
        st.error(
            f"The model bundles in `models/` could not be loaded ({exc}). "
            "Restore the directory from the repository, or re-export it with "
            "`python -m scripts.export_serving_models`."
        )
        st.stop()

    seed_end = pd.Timestamp(service.bundle.schema["seed_end"])
    next_ts = service.buffer.last_timestamp + STEP
    families = service.available_families

    st.markdown(
        f"Each session starts from the real readings up to **{seed_end:{TIME_FORMAT}}**, where the models' "
        "training data ends. Enter what the meter reads next; every model then forecasts the reading "
        "**60 minutes** after your latest one. Your readings live only in this browser session."
    )

    with st.container(border=True):
        with st.form("add_reading", border=False, enter_to_submit=True):
            c1, c2, c3 = st.columns([1.2, 1.4, 1], vertical_alignment="bottom")
            c1.number_input(
                "Next reading (Wh)",
                min_value=0.0,
                max_value=5000.0,
                value=60.0,
                step=10.0,
                key="reading_wh",
                help="Half of all recorded readings fall between 50 and 100 Wh (median 60); the highest is 1,080 Wh.",
            )
            c2.text_input(
                "Time of reading",
                value=f"{next_ts:{TIME_FORMAT}}",
                disabled=True,
                help="Readings must arrive in order, exactly 10 minutes apart.",
            )
            c3.form_submit_button(
                "Add reading", type="primary", icon=":material/add:", on_click=_add_reading, width="stretch"
            )
        left, right = st.columns([3, 1], vertical_alignment="center")
        n_added = st.session_state.n_added
        left.caption(
            f"{n_added} reading{'s' if n_added != 1 else ''} added this session. "
            f"Latest reading: {service.buffer.last_timestamp:{TIME_FORMAT}}."
        )
        right.button("Reset", icon=":material/restart_alt:", type="tertiary", on_click=_new_session, width="stretch")
        if st.session_state.flash:
            st.error(st.session_state.flash)

    selected = st.pills(
        "Models", families, selection_mode="multi", default=families, format_func=model_label, key="models"
    )
    if not selected:
        st.info("Select at least one model to see its forecast.")
        return

    try:
        with _predict_lock():
            results = service.predict(list(selected))
    except Exception as exc:
        st.error(f"Forecasting failed: {exc}")
        return

    st.altair_chart(_chart(service.buffer.committed_frame(), results, seed_end), width="stretch")

    ok = [r for r in results if "error" not in r]
    table = pd.DataFrame(
        {
            "Model": [model_label(r["model_family"]) for r in ok],
            "Forecast (Wh)": [r["prediction_wh"] for r in ok],
            "Forecast for": [f"{r['forecast_timestamp']:%a %d %b, %H:%M}" for r in ok],
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={"Forecast (Wh)": st.column_config.NumberColumn(format="%.1f")},
    )
    for r in results:
        if "error" in r:
            st.warning(f"{model_label(r['model_family'])} is not available ({r['error']}).")

    callout(
        "Forecasts are shown side by side, not ranked. In walk-forward evaluation no model family was "
        "shown to beat the classical references on both MAE and RMSE; see <b>Model comparison</b>."
    )
