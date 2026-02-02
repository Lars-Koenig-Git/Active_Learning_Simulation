#!/usr/bin/env python3

#-#####################################################################-#
# STAGE 6: PLOT RESULTS
# Uses the aggregated summary from stage 5: 03_results/stopping_rules_summary.csv
#-#####################################################################-#

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_PATH = Path(__file__).resolve().parent.parent
SUMMARY_CSV = PROJECT_PATH / "03_results" / "stopping_rules_summary.csv"
OUTPUT_PNG = PROJECT_PATH / "03_results" / "stopping_rules_faceted.png"


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Summary not found: {path}")
    return pd.read_csv(path)


def make_plot(df: pd.DataFrame) -> None:
    bands = ["<= 5% prevalence", "5%-10% prevalence", "> 10% prevalence"]
    rules = sorted(df["rule"].unique())

    fig, axes = plt.subplots(1, len(bands), figsize=(9, 3.6), sharey=True)

    for ax, band in zip(axes, bands):
        sub = df[df["band"] == band]
        for rule in rules:
            rsub = sub[sub["rule"] == rule]
            if rsub.empty:
                continue
            x = rsub["mean_cost"].values
            y = rsub["mean_recall"].values
            xerr = rsub["sd_cost"].fillna(0).values
            yerr = rsub["sd_recall"].fillna(0).values
            ax.errorbar(
                x,
                y,
                xerr=xerr,
                yerr=yerr,
                fmt="o",
                capsize=3,
                label=rule,
            )
        ax.set_title(band)
        ax.set_xlim(0, 1)
        ax.set_ylim(0.9, 1.0)
        ax.set_xlabel("Screening cost")
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("Recall (Sensitivity)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(rules))
    fig.tight_layout(rect=[0, 0.12, 1, 1])

    os.makedirs(OUTPUT_PNG.parent, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300)
    plt.close(fig)
    print(f"Wrote plot to {OUTPUT_PNG}")


def main() -> int:
    df = load_data(SUMMARY_CSV)
    make_plot(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
