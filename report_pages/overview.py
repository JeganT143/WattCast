"""Overview: what WattCast does, the headline results (computed from
results/), how the system fits together, and where to go next."""

import streamlit as st

from report_pages._style import page_header, source_note
from src.ui.content import DATASET
from src.ui.report_data import load_tier1_summary, load_verdicts, load_walk_forward_leaderboard, model_label

_PIPELINE_DOT = """
digraph {
  rankdir=TB; bgcolor="transparent"; nodesep=0.35; ranksep=0.32; newrank=true;
  node [shape=box, style="rounded,filled", fillcolor="#F1F5F9", color="#CBD5E1",
        fontname="Helvetica", fontsize=13, fontcolor="#1A202C", margin="0.22,0.1"];
  edge [color="#94A3B8", arrowsize=0.7];

  subgraph cluster_offline {
    label="Offline: data, training, evaluation"; labeljust="l"; fontname="Helvetica"; fontsize=12;
    fontcolor="#475569"; style="rounded,dashed"; color="#CBD5E1";
    raw   [label="UCI readings (10-min Appliances Wh)"];
    feat  [label="Leakage-safe features + train-only scaling"];
    model [label="5 Forecaster models: LR · RF · LSTM · GRU · CNN-LSTM", fillcolor="#DBE7F3", color="#1F4E79"];
    eval  [label="Walk-forward evaluation → results/"];
    reg   [label="MLflow registry (@champion)"];
  }
  subgraph cluster_online {
    label="Online: serving"; labeljust="l"; fontname="Helvetica"; fontsize=12;
    fontcolor="#475569"; style="rounded,dashed"; color="#CBD5E1";
    bund [label="Exported bundles (models/)"];
    app  [label="Streamlit app", fillcolor="#DBE7F3", color="#1F4E79"];
    api  [label="FastAPI service", fillcolor="#DBE7F3", color="#1F4E79"];
  }
  raw -> feat -> model;
  model -> eval; model -> reg;
  reg -> bund [label=" export", fontname="Helvetica", fontsize=11, fontcolor="#64748B"];
  bund -> app; bund -> api;
}
"""

_PAGES = [
    ("Live forecast", "Enter meter readings and compare each model's one-hour-ahead forecast."),
    ("Model comparison", "Walk-forward results, the pre-registered verdicts and the held-out test check."),
    ("Data & features", "The dataset, what exploration showed, and the leakage-safe feature set."),
    ("Method & limitations", "How the evaluation was designed, how serving works, and what the results do not show."),
]


def render() -> None:
    page_header(
        "WattCast",
        "One-hour-ahead forecasts of household appliance energy use, "
        "evaluated under a pre-registered, leakage-safe protocol.",
        kicker="Appliance energy forecasting",
    )

    st.markdown(
        "WattCast predicts how much energy a household's appliances will use **60 minutes from now**, "
        "from the most recent 10-minute meter readings. It compares naive baselines, linear regression, "
        "a random forest and three deep sequence models (LSTM, GRU, CNN-LSTM), all behind one shared "
        "interface. The five learned models are served live on the next page and through a REST API."
    )

    board = load_walk_forward_leaderboard().set_index("model")
    best_mae = board["mean_mae"].idxmin()
    best_rmse = board["mean_rmse"].idxmin()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecast horizon", "60 min", help="6 steps of 10 minutes ahead")
    c2.metric("Models served", "5", help="Linear regression, random forest, LSTM, GRU, CNN-LSTM")
    c3.metric(
        "Lowest mean MAE", f"{board.at[best_mae, 'mean_mae']:.1f} Wh",
        help=f"{model_label(best_mae)}, mean over 8 walk-forward folds",
    )
    c4.metric(
        "Lowest mean RMSE", f"{board.at[best_rmse, 'mean_rmse']:.1f} Wh",
        help=f"{model_label(best_rmse)}, mean over 8 walk-forward folds",
    )

    st.subheader("Key findings", anchor=False)
    deep = board[board["group"] == "Deep sequence"]
    lr = board.loc["linear_regression"]
    verdicts = load_verdicts()
    tier1 = load_tier1_summary()
    n_not_shown = int((verdicts["verdict"] == "not shown").sum())
    st.markdown(
        f"- **Deep models make smaller typical errors.** {model_label(best_mae)} averages "
        f"{board.at[best_mae, 'mean_mae']:.2f} Wh MAE over 8 walk-forward folds, against "
        f"{lr['mean_mae']:.2f} Wh for linear regression.\n"
        f"- **They also make bigger large errors.** Every deep model's mean RMSE "
        f"({deep['mean_rmse'].min():.2f}–{deep['mean_rmse'].max():.2f} Wh) is above linear regression's "
        f"{lr['mean_rmse']:.2f} Wh.\n"
        f"- **No model is shown better.** The pre-registered rule needs lower MAE *and* RMSE for every "
        f"training seed; {n_not_shown} of {len(verdicts)} deep-vs-classical comparisons are \"not shown\".\n"
        f"- **The held-out test agrees.** In a one-time test-set check, the LSTM beat linear regression's MAE bar "
        f"with {sum(r['mae_pass'] for r in tier1['runs'])} of 3 seeds and its RMSE bar with "
        f"{sum(r['rmse_pass'] for r in tier1['runs'])} of 3.\n"
        f"- **Linear regression is the primary serving model** (lowest mean RMSE); all five learned models "
        f"are served for side-by-side comparison."
    )
    source_note("results/walk_forward_summary.json, results/final_evaluation.json")

    st.subheader("How it works", anchor=False)
    st.graphviz_chart(_PIPELINE_DOT, width="stretch")
    st.caption(
        f"Data: {DATASET['name']} ({DATASET['authors']}): {DATASET['n_rows']:,} readings from one house, "
        f"{DATASET['start']} to {DATASET['end']}. Training, tracking and export run offline; the app and the "
        "API only load the exported bundles."
    )

    st.subheader("Explore", anchor=False)
    cols = st.columns(2)
    for i, (title, text) in enumerate(_PAGES):
        with cols[i % 2].container(border=True):
            st.markdown(f"**{title}**  \n{text}")
