#!/usr/bin/env python3

#-#####################################################################-#
# STAGE 5: BUILD SUMMARY STATISTICS ####
#-#####################################################################-#

import os
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_PATH = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_PATH / "03_results" / "stopping_rules_summary_key_parallel.csv"
OUTPUT_CSV = PROJECT_PATH / "03_results" / "stopping_rules_summary.csv"


#-#####################################################################-#
# HELPERS ####
#-#####################################################################-#
def band_label(p: float) -> str:
    """Bucket prevalence into three bands."""
    if pd.isna(p) or p <= 0.05:
        return "<= 5% prevalence"
    if p <= 0.10:
        return "5%-10% prevalence"
    return "> 10% prevalence"


def aggregate_for_rule(df_band: pd.DataFrame, rule: str) -> dict:
    """Aggregate metrics for one rule within one band.

    Args:
        df_band: Filtered dataframe for a prevalence band.
        rule: Rule name (A/B/C).
    """
    # Pull core metrics and diagnostics
    cost = df_band[f"{rule}_screening_cost"].astype(float)
    recall = df_band[f"{rule}_recall"].astype(float)
    breakout = df_band[f"{rule}_stopped_by_breakout"].astype(bool)
    data_cut_missing = df_band[f"{rule}_data_cut_met_without_all_keys"].astype(bool)
    gain_from_keys = pd.to_numeric(df_band.get(f"{rule}_sensitivity_gain_from_keys"), errors="coerce")
    keys_missed = pd.to_numeric(df_band.get(f"{rule}_keys_missing_at_data_cut"), errors="coerce")

    mean_cost = float(cost.mean()) if len(cost) else np.nan
    sd_cost = float(cost.std(ddof=1)) if len(cost) > 1 else 0.0
    mean_recall = float(recall.mean()) if len(recall) else np.nan
    sd_recall = float(recall.std(ddof=1)) if len(recall) > 1 else 0.0

    prop_recall_ge_95 = float((recall >= 0.95).mean()) if len(recall) else np.nan
    prop_recall_eq_100 = float((recall >= 0.999999).mean()) if len(recall) else np.nan
    prop_breakout_used = float(breakout.mean()) if len(breakout) else np.nan
    prop_data_cut_missing_keys = float(data_cut_missing.mean()) if len(data_cut_missing) else np.nan

    n_breakout_runs = int(breakout.sum()) if len(breakout) else 0
    n_runs_data_cut_missing_keys = int(data_cut_missing.sum()) if len(data_cut_missing) else 0

    mean_recall_gain_from_keys = (
        float(gain_from_keys.dropna().mean()) if gain_from_keys is not None and (~gain_from_keys.isna()).any() else 0.0
    )
    avg_keys_missed_when_missing = (
        float(keys_missed[data_cut_missing].mean()) if n_runs_data_cut_missing_keys > 0 else 0.0
    )

    return {
        "mean_cost": mean_cost,
        "sd_cost": sd_cost,
        "mean_recall": mean_recall,
        "sd_recall": sd_recall,
        "n_>=95": prop_recall_ge_95,
        "n_=100": prop_recall_eq_100,
        "prop_breakout_used": prop_breakout_used,
        "prop_data_cut_missing_keys": prop_data_cut_missing_keys,
        "mean_recall_gain_from_keys": mean_recall_gain_from_keys,
        "n_breakout_runs": n_breakout_runs,
        "n_runs_data_cut_missing_keys": n_runs_data_cut_missing_keys,
        "avg_keys_missed_when_missing": avg_keys_missed_when_missing,
    }


#-#####################################################################-#
# MAIN EXECUTION PIPELINE ####
#-#####################################################################-#
def main() -> int:
    """Build the summary table for stopping rules."""
    # Load per-dataset stopping stats
    if not INPUT_CSV.exists():
        print(f"Input not found: {INPUT_CSV}")
        return 1
    df = pd.read_csv(INPUT_CSV)

    # Bucket datasets by prevalence band
    df["__band_label"] = df["training_prevalence_estimate"].apply(band_label)

    rows = []
    for band in ["<= 5% prevalence", "5%-10% prevalence", "> 10% prevalence"]:
        df_band = df[df["__band_label"] == band].copy()
        if df_band.empty:
            continue

        n_datasets = int(len(df_band))
        avg_prev = float(df_band["training_prevalence_estimate"].mean()) if n_datasets else np.nan

        for rule in ["A", "B", "C"]:
            stats = aggregate_for_rule(df_band, rule)
            rows.append(
                {
                    "band": band,
                    "rule": rule,
                    **stats,
                    "n_datasets": n_datasets,
                    "avg_training_prevalence": avg_prev,
                }
            )

    out_df = pd.DataFrame(rows)

    # Stable ordering for readability
    band_order = {"<= 5% prevalence": 0, "5%-10% prevalence": 1, "> 10% prevalence": 2}
    out_df["__band_order"] = out_df["band"].map(band_order)
    out_df["__rule_order"] = out_df["rule"].map({"A": 0, "B": 1, "C": 2})
    out_df = (
        out_df.sort_values(["__band_order", "__rule_order"])
        .drop(columns=["__band_order", "__rule_order"])
        .reset_index(drop=True)
    )

    os.makedirs(OUTPUT_CSV.parent, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV} with {len(out_df)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
