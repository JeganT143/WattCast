"""Model Comparison page: the registered 8-fold walk-forward leaderboard,
verdicts, and per-fold detail, all read directly from
results/walk_forward_summary.json and results/walk_forward_records.jsonl
via src/ui/report_data.py. The three walk-forward images already produced
in an earlier stage are embedded as-is.
"""

import streamlit as st

from src.ui.report_data import (
    available_images,
    load_per_fold_table,
    load_verdicts,
    load_walk_forward_leaderboard,
)

_VERDICT_EXPLANATION = (
    "**No model family has been shown to decisively outperform the others.** "
    "The pre-registered 8-fold walk-forward comparison required a deep model "
    "(LSTM/GRU/CNN-LSTM) to beat each reference (linear_regression, random_forest) "
    "on **both** MAE and RMSE, for every seed, to count as \"shown better\" — a "
    "mixed result (better on one metric, worse on the other) is registered as "
    "\"not shown\", not as a tie or as evidence of equivalence.\n\n"
    "All six comparisons (3 deep models x 2 references) came back **\"not shown\"**: "
    "in every pair, the deep model's MAE ratio was consistently better (<= 0.99) "
    "while its RMSE ratio was consistently worse (>= 1.01) than the reference. "
    "These predictions are shown here for side-by-side comparison only, not to "
    "declare a winner."
)


def render() -> None:
    st.title("WattCast — Model Comparison")

    st.subheader("Walk-forward leaderboard (8 folds, mean over folds)")
    st.caption("Source: results/walk_forward_summary.json, summary.per_model_means")
    leaderboard = load_walk_forward_leaderboard()
    st.dataframe(
        leaderboard.rename(
            columns={"mean_mae": "Mean MAE", "mean_rmse": "Mean RMSE", "mean_mape": "Mean MAPE"}
        ),
        hide_index=True,
    )

    st.subheader('The "not shown" verdicts')
    st.markdown(_VERDICT_EXPLANATION)
    st.caption("Source: results/walk_forward_summary.json, summary.verdicts")
    verdicts = load_verdicts()
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
    )

    st.subheader("Walk-forward diagnostics")
    images = available_images()
    st.image(str(images["walk_forward_folds"]), caption="The 8 expanding-window folds.")
    st.image(
        str(images["walk_forward_means"]),
        caption="Per-model mean MAE/RMSE across the 8 folds.",
    )
    st.image(
        str(images["walk_forward_ratios"]),
        caption="Per-fold MAE/RMSE ratios of each deep model against each reference.",
    )
    st.image(
        str(images["final_predictions_h6"]),
        caption="Final h=6 predictions (registered final evaluation).",
    )

    with st.expander("Per-fold, per-model detail (mean over seeds)"):
        st.caption("Source: results/walk_forward_records.jsonl, grouped by (fold, model)")
        st.dataframe(load_per_fold_table(), hide_index=True)
