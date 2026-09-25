"""Model Comparison page: the registered 8-fold walk-forward leaderboard,
verdicts, and per-fold detail, all read directly from
results/walk_forward_summary.json and results/walk_forward_records.jsonl
via src/ui/report_data.py. The three walk-forward images already produced
in an earlier stage are embedded as-is.
"""

import streamlit as st

from report_pages._style import callout, page_header, source_caption
from src.ui.report_data import (
    available_images,
    load_per_fold_table,
    load_verdicts,
    load_walk_forward_leaderboard,
)

_VERDICT_EXPLANATION = (
    "<strong>No model family has been shown to decisively outperform the others.</strong> "
    "The pre-registered 8-fold walk-forward comparison required a deep model "
    "(LSTM/GRU/CNN-LSTM) to beat each reference (linear_regression, random_forest) "
    "on <strong>both</strong> MAE and RMSE, for every seed, to count as "
    '"shown better" &mdash; a mixed result (better on one metric, worse on the '
    'other) is registered as "not shown", not as a tie or as evidence of '
    "equivalence."
    "<br><br>"
    'All six comparisons (3 deep models &times; 2 references) came back '
    '<strong>"not shown"</strong>: in every pair, the deep model\'s MAE ratio was '
    "consistently better (&le; 0.99) while its RMSE ratio was consistently worse "
    "(&ge; 1.01) than the reference. These predictions are shown here for "
    "side-by-side comparison only, not to declare a winner."
)


def _format_ratios(ratios: list[float]) -> str:
    return " / ".join(f"{r:.3f}" for r in ratios)


def render() -> None:
    page_header("Model Comparison", "The registered 8-fold walk-forward results")

    st.markdown("#### Leaderboard (mean over 8 folds)")
    source_caption("results/walk_forward_summary.json · summary.per_model_means")
    leaderboard = load_walk_forward_leaderboard().round(2)
    st.dataframe(
        leaderboard.rename(
            columns={"mean_mae": "Mean MAE", "mean_rmse": "Mean RMSE", "mean_mape": "Mean MAPE (%)"}
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown('#### The "not shown" verdicts')
    callout(_VERDICT_EXPLANATION)
    source_caption("results/walk_forward_summary.json · summary.verdicts")
    verdicts = load_verdicts().copy()
    verdicts["mean_mae_ratios"] = verdicts["mean_mae_ratios"].apply(_format_ratios)
    verdicts["mean_rmse_ratios"] = verdicts["mean_rmse_ratios"].apply(_format_ratios)
    st.dataframe(
        verdicts.rename(
            columns={
                "model": "Model",
                "reference": "Reference",
                "verdict": "Verdict",
                "mean_mae_ratios": "MAE ratios (seeds 42/43/44)",
                "mean_rmse_ratios": "RMSE ratios (seeds 42/43/44)",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("#### Walk-forward diagnostics")
    images = available_images()
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.image(str(images["walk_forward_folds"]), caption="The 8 expanding-window folds")
    with row1_col2:
        st.image(
            str(images["walk_forward_means"]),
            caption="Per-model mean MAE/RMSE across the 8 folds",
        )
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.image(
            str(images["walk_forward_ratios"]),
            caption="Per-fold MAE/RMSE ratios, deep models vs. references",
        )
    with row2_col2:
        st.image(
            str(images["final_predictions_h6"]),
            caption="Final h=6 predictions (registered final evaluation)",
        )

    with st.expander("Per-fold, per-model detail (mean over seeds)"):
        source_caption("results/walk_forward_records.jsonl · grouped by (fold, model)")
        st.dataframe(load_per_fold_table().round(2), hide_index=True, use_container_width=True)
