"""Smoke tests for the Streamlit app: every page renders without an
exception, and the Live forecast page accepts a reading and forecasts
with all five committed models."""

import pytest
from streamlit.testing.v1 import AppTest

from config.paths import PROJECT_ROOT

PAGES = ["overview", "live_inference", "model_comparison", "data_features", "limitations"]


def _page_app(module: str) -> AppTest:
    script = (
        "from report_pages._style import inject_base_styles\n"
        f"from report_pages import {module}\n"
        "inject_base_styles()\n"
        f"{module}.render()\n"
    )
    return AppTest.from_string(script, default_timeout=120)


@pytest.mark.parametrize("module", PAGES)
def test_page_renders_without_exceptions(module):
    at = _page_app(module).run()
    assert not at.exception, at.exception


def test_entrypoint_runs():
    at = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception, at.exception


def test_live_forecast_adds_a_reading_and_forecasts_all_models():
    at = _page_app("live_inference").run()
    assert at.dataframe[0].value["Model"].tolist() == ["Linear regression", "Random forest", "LSTM", "GRU", "CNN-LSTM"]

    at.number_input(key="reading_wh").set_value(250.0)
    at.button[0].click().run()  # the form's "Add reading" submit button
    assert not at.exception, at.exception
    assert not at.error

    service = at.session_state["service"]
    frame = service.buffer.committed_frame()
    assert str(frame["date"].iloc[-1]) == "2016-04-30 00:00:00"
    assert frame["Appliances"].iloc[-1] == 250.0
    assert at.session_state["n_added"] == 1

    table = at.dataframe[0].value
    assert len(table) == 5
    assert table["Forecast for"].eq("Sat 30 Apr, 01:00").all()
