#!/usr/bin/env python3

#-#####################################################################-#
# STAGE 4: ANALYZE STOPPING RULES ####
#-#####################################################################-#

import glob
import math
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd

#-#####################################################################-#
# CONFIGURATION ####
#-#####################################################################-#
PROJECT_PATH = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_PATH / "02_data" / "03_screened_data"
OUTPUT_DIR = PROJECT_PATH / "03_results"
OUTPUT_CSV = OUTPUT_DIR / "stopping_rules_summary_key_parallel.csv"

RULESETS = {
    "A": {"time_cut": 0.35, "data_cut": 0.07, "breakout_cut": 0.15},
    "B": {"time_cut": 0.25, "data_cut": 0.05, "breakout_cut": 0.10},
    "C": {"time_cut": 0.15, "data_cut": 0.03, "breakout_cut": 0.10},
}


#-#####################################################################-#
# HELPERS ####
#-#####################################################################-#
def select_band_by_prevalence(p: float) -> str:
    """Pick ruleset band from prevalence estimate."""
    if p is None or np.isnan(p) or p <= 0.05:
        return "A"
    if p <= 0.10:
        return "B"
    return "C"


def get_order_column(df: pd.DataFrame) -> str:
    """Return the screening order column name (assumed present)."""
    return "sequential_screening_order"


def parse_algorithm_from_filename(path: str) -> str:
    """Extract algorithm token from filename (best-effort)."""
    name = os.path.basename(path)
    for alg in ["ELAS_Ultra", "ELAS_Heavy", "LR_LaBSE", "LR_SBERT"]:
        if alg in name:
            return alg
    return ""


#-#####################################################################-#
# CORE LOGIC ####
#-#####################################################################-#
def compute_stop_for_ruleset(df: pd.DataFrame, ruleset: dict) -> dict:
    """Apply one stopping ruleset and return metrics.

    Args:
        df: Simulation results with status/label/is_key_study columns.
        ruleset: Dict with time_cut, data_cut, breakout_cut.
    """
    # Partition training vs non-training and gather counts
    order_col = get_order_column(df)
    status = df["status"]
    is_training = status.eq("training")

    train_df = df[is_training].copy()
    train_size = int(len(train_df))
    train_relevant = int((train_df["label"] == 1).sum())
    train_keys_found = int(((train_df["is_key_study"] == 1) & (train_df["label"] == 1)).sum())

    nontrain_df = df[~is_training].copy().sort_values(order_col, kind="mergesort").reset_index(drop=True)
    nontrain_size = int(len(nontrain_df))

    total_size = int(len(df))
    total_relevant = int((df["label"] == 1).sum())
    total_relevant_nontraining = int(((df["label"] == 1) & (~is_training)).sum())
    total_keys = int((df["is_key_study"] == 1).sum())

    time_cut_n = int(math.ceil(ruleset["time_cut"] * nontrain_size))
    data_cut_n = int(math.ceil(ruleset["data_cut"] * nontrain_size))
    breakout_cut_n = int(math.ceil(ruleset["breakout_cut"] * nontrain_size))

    consec_irrel_since_last_rel = 0
    keys_found_so_far = train_keys_found
    relevant_seen_cumulative = train_relevant

    stop_index_nontraining = None
    stop_mode = None

    first_data_cut_met_index = None
    recall_at_data_cut_met = np.nan
    keys_found_at_data_cut_met = None
    keys_missing_at_data_cut = None

    # Sweep over ordered non-training records until a stop condition is met
    for i, row in nontrain_df.iterrows():
        is_rel = (row["label"] == 1)
        is_key = (row["is_key_study"] == 1) and is_rel

        if is_rel:
            consec_irrel_since_last_rel = 0
            relevant_seen_cumulative += 1
            if is_key:
                keys_found_so_far += 1
        else:
            consec_irrel_since_last_rel += 1

        progressed = i + 1
        time_reached = progressed >= time_cut_n

        if not time_reached:
            if consec_irrel_since_last_rel >= breakout_cut_n and keys_found_so_far >= total_keys:
                stop_index_nontraining = progressed
                stop_mode = "breakout"
                break
        else:
            if consec_irrel_since_last_rel >= data_cut_n and first_data_cut_met_index is None:
                first_data_cut_met_index = progressed
                recall_at_data_cut_met = relevant_seen_cumulative / total_relevant if total_relevant > 0 else np.nan
                keys_found_at_data_cut_met = keys_found_so_far
                keys_missing_at_data_cut = max(0, total_keys - (keys_found_at_data_cut_met or 0))

            if consec_irrel_since_last_rel >= data_cut_n and keys_found_so_far >= total_keys:
                stop_index_nontraining = progressed
                stop_mode = "data_cut"
                break

    # Default stop at end if no earlier rule triggered
    if stop_index_nontraining is None:
        stop_index_nontraining = nontrain_size
        stop_mode = "end"

    screened_total_including_training = train_size + stop_index_nontraining
    screening_cost = screened_total_including_training / total_size if total_size > 0 else np.nan
    relevant_found_total = relevant_seen_cumulative
    recall = relevant_found_total / total_relevant if total_relevant > 0 else np.nan

    data_cut_met_without_all_keys = (
        first_data_cut_met_index is not None and (keys_found_at_data_cut_met or 0) < total_keys
    )

    sensitivity_gain_from_keys = np.nan
    if data_cut_met_without_all_keys and total_relevant > 0:
        keys_found_at_end = keys_found_so_far
        keys_found_after_data_cut = max(
            0, (keys_found_at_end or 0) - (keys_found_at_data_cut_met or 0)
        )
        sensitivity_gain_from_keys = keys_found_after_data_cut / total_relevant

    return {
        "stop_count_nontraining": stop_index_nontraining,
        "screening_cost": screening_cost,
        "recall": recall,
        "all_keys_found": (keys_found_so_far >= total_keys),
        "total_relevant": total_relevant,
        "total_abstracts": total_size,
        "total_relevant_in_screening_set": total_relevant_nontraining,
        "training_size": train_size,
        "training_relevant": train_relevant,
        "stopped_by_breakout": (stop_mode == "breakout"),
        "data_cut_met_without_all_keys": data_cut_met_without_all_keys,
        "sensitivity_gain_from_keys": sensitivity_gain_from_keys,
        "keys_missing_at_data_cut": keys_missing_at_data_cut,
    }


def analyze_file(path: str) -> dict | None:
    """Run stopping rules for a single simulation result.

    Args:
        path: CSV path from Stage 3 output.
    """
    df = pd.read_csv(path)

    # Estimate prevalence band from training set and keys
    train_mask = df["status"].eq("training")
    train_size = int(train_mask.sum())
    total_keys = int((df.get("is_key_study", 0) == 1).sum())
    denom = train_size + total_keys
    prev_est = ((total_keys + 1) / denom) if denom > 0 else np.nan

    selected_band = select_band_by_prevalence(prev_est)
    results_by_rule = {rule_name: compute_stop_for_ruleset(df, params) for rule_name, params in RULESETS.items()}
    sel = results_by_rule[selected_band]

    # Build summary row for this dataset
    base_name = os.path.basename(path)
    algorithm = parse_algorithm_from_filename(path)

    row = {
        "dataset_file": base_name,
        "algorithm": algorithm,
        "total_abstracts": sel["total_abstracts"],
        "total_relevant": sel["total_relevant"],
        "total_relevant_in_screening_set": sel["total_relevant_in_screening_set"],
        "training_size": sel["training_size"],
        "training_relevant": sel["training_relevant"],
        "training_prevalence_estimate": prev_est,
        "band_selected_by_estimate": selected_band,
        "selected_screening_cost": sel["screening_cost"],
        "selected_recall": sel["recall"],
        "selected_all_keys_found": sel["all_keys_found"],
    }

    for rule_name in ["A", "B", "C"]:
        r = results_by_rule[rule_name]
        row[f"{rule_name}_screening_cost"] = r["screening_cost"]
        row[f"{rule_name}_recall"] = r["recall"]
        row[f"{rule_name}_all_keys_found"] = r["all_keys_found"]
        row[f"{rule_name}_stopped_by_breakout"] = r["stopped_by_breakout"]
        row[f"{rule_name}_data_cut_met_without_all_keys"] = r["data_cut_met_without_all_keys"]
        row[f"{rule_name}_sensitivity_gain_from_keys"] = r["sensitivity_gain_from_keys"]
        row[f"{rule_name}_keys_missing_at_data_cut"] = r["keys_missing_at_data_cut"]

    return row


#-#####################################################################-#
# MAIN EXECUTION PIPELINE ####
#-#####################################################################-#
def main() -> int:
    """Parallelize stopping-rule analysis over all simulation outputs."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(glob.glob(str(INPUT_DIR / "*.csv")))

    # Choose a modest worker count (leave a few cores free)
    cpu_total = os.cpu_count() or 1
    workers = max(1, cpu_total - 4)

    # Map-reduce over all files
    rows: list[dict] = []
    if files:
        with mp.get_context("spawn").Pool(processes=workers) as pool:
            for res in pool.imap_unordered(analyze_file, files, chunksize=8):
                if res is not None:
                    rows.append(res)

    if rows:
        pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# ###-#
