"""Exploratory, descriptive plot of the h=6 test predictions. Two fixed windows chosen by
position only (the first and the last 1,008 rows of the test partition, 7 days each); no
metrics are computed. It cannot change the registered verdict.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

WINDOW = 1008
SEEDS = (42, 43, 44)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    df = pd.read_csv(args.predictions, parse_dates=["date"])
    assert list(df.columns) == ["date", "actual", "pred_seed_42", "pred_seed_43", "pred_seed_44"] and len(df) >= 2 * WINDOW and df.notna().all().all()

    windows = [("first 7 days of the test partition", df.iloc[:WINDOW]), ("last 7 days of the test partition", df.iloc[-WINDOW:])]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharey=True)
    for ax, (label, w) in zip(axes, windows):
        ax.plot(w["date"], w["actual"], color="black", linewidth=1.3, label="actual")
        for seed, color in zip(SEEDS, ("tab:blue", "tab:orange", "tab:green")):
            ax.plot(w["date"], w["pred_seed_%d" % seed], color=color, linewidth=0.8, alpha=0.85, label="LSTM seed %d" % seed)
        ax.set_title("%s: %s to %s (%d rows)" % (label, w["date"].iloc[0], w["date"].iloc[-1], len(w)), fontsize=10)
        ax.set_ylabel("target_t6 (Wh)"); ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", ncol=4, fontsize=9); axes[1].set_xlabel("date")
    fig.suptitle("LSTM test predictions vs actual, h=6 (exploratory, descriptive only; the registered Tier 1 verdict is unchanged)", fontsize=11)
    fig.tight_layout(); fig.savefig(args.out, dpi=110); plt.close(fig)
    print("wrote", args.out); print("rows in file:", len(df)); print("windows:", [(lbl, str(w["date"].iloc[0]), str(w["date"].iloc[-1])) for lbl, w in windows])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
