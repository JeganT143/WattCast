"""Model comparison: the 8-fold walk-forward leaderboard, the six
pre-registered verdicts, the one-shot held-out test check, and the
diagnostic figures — all read from results/ via src/ui/report_data.py."""

import altair as alt
import pandas as pd
import streamlit as st

from report_pages._style import MODEL_COLORS, callout, page_header, source_note
from src.ui.report_data import (
    available_images,
    load_per_fold_table,
    load_tier1_summary,
    load_verdicts,
    load_walk_forward_leaderboard,
    model_label,
)

_METRICS = {
    "MAE": ("mean_mae", "Mean absolute error (Wh)"),
    "RMSE": ("mean_rmse", "Root mean squared error (Wh)"),
    "MAPE": ("mean_mape", "Mean absolute percentage error (%)"),
}


def _ratios(values: list[float]) -> str:
    return " / ".join(f"{v:.3f}" for v in values)


def _leaderboard(board: pd.DataFrame) -> None:
    metric = st.segmented_control("Metric", list(_METRICS), default="MAE", key="leaderboard_metric") or "MAE"
    column, title = _METRICS[metric]
    chart = (
        alt.Chart(board)
        .mark_bar(cornerRadiusEnd=3, height=18)
        .encode(
            x=alt.X(f"{column}:Q", title=title),
            y=alt.Y("label:N", sort=board.sort_values(column)["label"].tolist(), title=None),
            color=alt.Color(
                "model:N",
                scale=alt.Scale(domain=list(MODEL_COLORS), range=list(MODEL_COLORS.values())),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Model"),
                alt.Tooltip("group:N", title="Type"),
                alt.Tooltip(f"{column}:Q", title=metric, format=".2f"),
            ],
        )
        .properties(height=alt.Step(28))
    )
    labels = chart.mark_text(align="left", dx=4, color="#334155").encode(text=alt.Text(f"{column}:Q", format=".1f"))
    st.altair_chart(chart + labels, width="stretch")

    st.dataframe(
        board[["label", "group", "mean_mae", "mean_rmse", "mean_mape"]],
        hide_index=True,
        width="stretch",
        column_config={
            "label": "Model",
            "group": "Type",
            "mean_mae": st.column_config.NumberColumn("MAE (Wh)", format="%.2f"),
            "mean_rmse": st.column_config.NumberColumn("RMSE (Wh)", format="%.2f"),
            "mean_mape": st.column_config.NumberColumn("MAPE (%)", format="%.2f", help="Readings below 30 Wh excluded"),
        },
    )
    source_note("results/walk_forward_summary.json (per-model means)")


def _verdicts() -> None:
    verdicts = load_verdicts()
    order = {model: i for i, model in enumerate(MODEL_COLORS)}
    verdicts = verdicts.sort_values(["model", "reference"], key=lambda col: col.map(order))
    table = pd.DataFrame(
        {
            "Deep model": verdicts["model"].map(model_label),
            "Reference": verdicts["reference"].map(model_label),
            "Verdict": verdicts["verdict"].str.capitalize(),
            "MAE ratio (seeds 42 / 43 / 44)": verdicts["mean_mae_ratios"].map(_ratios),
            "RMSE ratio (seeds 42 / 43 / 44)": verdicts["mean_rmse_ratios"].map(_ratios),
        }
    )
    st.dataframe(table, hide_index=True, width="stretch")
    source_note("results/walk_forward_summary.json (verdicts)")


def _test_check() -> None:
    tier1 = load_tier1_summary()
    table = pd.DataFrame(
        {
            "Seed": [r["seed"] for r in tier1["runs"]],
            "Test MAE (Wh)": [r["test_mae"] for r in tier1["runs"]],
            "MAE bar met": ["Yes" if r["mae_pass"] else "No" for r in tier1["runs"]],
            "Test RMSE (Wh)": [r["test_rmse"] for r in tier1["runs"]],
            "RMSE bar met": ["Yes" if r["rmse_pass"] else "No" for r in tier1["runs"]],
        }
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE bar", f"≤ {tier1['threshold_mae']:.2f} Wh", help=f"0.99 × linear regression ({tier1['ref_mae']:.2f})")
    c2.metric("RMSE bar", f"≤ {tier1['threshold_rmse']:.2f} Wh", help=f"0.99 × linear regression ({tier1['ref_rmse']:.2f})")
    c3.metric("Result", "Shown" if tier1["shown"] else "Not shown")
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Test MAE (Wh)": st.column_config.NumberColumn(format="%.2f"),
            "Test RMSE (Wh)": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    source_note("results/final_evaluation.json")


def render() -> None:
    page_header(
        "Model comparison",
        "How seven models performed when forecasting 60 minutes ahead across eight consecutive weeks.",
    )

    st.markdown(
        "**Walk-forward evaluation.** Each of 8 folds trains on every reading before a 7-day window "
        "(1,008 readings; windows cover 1 Mar – 25 Apr 2016) and forecasts that window. Every fold refits its own "
        "train-only scaler. Deep models are trained with 3 seeds and averaged. Lower is better for all metrics."
    )

    st.subheader("Leaderboard", anchor=False)
    _leaderboard(load_walk_forward_leaderboard())

    st.subheader("Pre-registered verdicts", anchor=False)
    callout(
        "The rule was fixed before any fold was run: a deep model is <b>shown better</b> than a reference only "
        "if its mean error ratio is ≤ 0.99 on <b>both</b> MAE and RMSE for <b>every</b> seed (≥ 1.01 on both "
        "for <b>shown worse</b>). Everything else is <b>not shown</b>, which is not evidence of equivalence. "
        "All six comparisons are mixed the same way: deep models have lower MAE (ratio &lt; 1) but higher RMSE "
        "(ratio &gt; 1)."
    )
    _verdicts()

    st.subheader("Held-out test check", anchor=False)
    st.markdown(
        "Before the walk-forward study, the tuned LSTM was scored **once per seed** on the held-out test "
        "period (30 Apr – 27 May 2016) against a pre-registered bar: at least 1% lower MAE *and* RMSE than "
        "linear regression's test score from the earlier baseline study, for every seed."
    )
    _test_check()

    st.subheader("Diagnostics", anchor=False)
    images = available_images()
    tabs = st.tabs(["Per-fold ratios", "Fold layout", "Mean errors", "LSTM test predictions", "Per-fold table"])
    captions = [
        ("walk_forward_ratios", "Deep model ÷ reference error, per fold (line: seed mean; band: seed range)."),
        ("walk_forward_folds", "The 8 expanding training windows and their 7-day evaluation windows."),
        ("walk_forward_means", "Mean MAE and RMSE over the 8 folds."),
        ("final_predictions_h6", "LSTM forecasts vs actual readings in the first and last test weeks (descriptive only)."),
    ]
    for tab, (key, caption) in zip(tabs, captions):
        with tab:
            st.image(str(images[key]), caption=caption, width="stretch")
    with tabs[-1]:
        per_fold = load_per_fold_table()
        per_fold["model"] = per_fold["model"].map(model_label)
        st.dataframe(
            per_fold,
            hide_index=True,
            width="stretch",
            height=420,
            column_config={
                "fold": st.column_config.NumberColumn("Fold", format="%d"),
                "model": "Model",
                "mae": st.column_config.NumberColumn("MAE (Wh)", format="%.2f"),
                "rmse": st.column_config.NumberColumn("RMSE (Wh)", format="%.2f"),
                "mape": st.column_config.NumberColumn("MAPE (%)", format="%.2f"),
            },
        )
        source_note("results/walk_forward_records.jsonl (deep models averaged over seeds)")
