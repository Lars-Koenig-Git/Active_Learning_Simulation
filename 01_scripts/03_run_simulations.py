#!/usr/bin/env python3

#-#####################################################################-#
# STAGE 3: RUN SIMULATIONS ####
#-#####################################################################-#

import glob
import os
import random
import time

# Core data and ML deps
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# ASReview components
import asreview as asr
from asreview.models.balancers import Balanced
from asreview.models.classifiers import Logistic, SVM
from asreview.models.feature_extractors import Tfidf
from asreview.models.queriers import Max
from asreviewcontrib.dory.feature_extractors.sentence_transformer_embeddings import SBERT, LaBSE, MXBAI
#-#####################################################################-#
# PROJECT CONFIGURATION ####
#-#####################################################################-#
PROJECT_PATH = str(Path(__file__).resolve().parent.parent)
TRAINING_DATASETS_DIR = os.path.join(PROJECT_PATH, "02_data", "02_processed_input")
OUTPUT_DIR = os.path.join(PROJECT_PATH, "02_data", "03_screened_data")
SUMMARY_DIR = os.path.join(PROJECT_PATH, "02_data", "00_meta_data")

RANDOM_SEED = 42
TRAINING_MODE = 1  # 1: only marked training samples; 2: all sampled records
SINGLE_ALGORITHM = "LR_SBERT"  # set None to run all
USE_GPU = False  # default CPU; set True to allow GPU if available

#-#####################################################################-#
# ALGORITHM DEFINITIONS ####
#-#####################################################################-#
ALGORITHMS = {
    "LR_SBERT": {
        "feature_extractor": SBERT(),
        "classifier": Logistic(),
        "balancer": Balanced(),
        "querier": Max(),
        "description": "SBERT + Logistic",
    }
    ,
    "ELAS_Ultra": {
        "feature_extractor": Tfidf(),
        "classifier": SVM(),
        "balancer": Balanced(),
        "querier": Max(),
        "description": "TF-IDF + SVM",
    },
    "ELAS_Heavy": {
        "feature_extractor": MXBAI(),
        "classifier": SVM(),
        "balancer": Balanced(),
        "querier": Max(),
        "description": "MXBAI + SVM",
    },
    "LR_LaBSE": {
        "feature_extractor": LaBSE(),
        "classifier": Logistic(),
        "balancer": Balanced(),
        "querier": Max(),
        "description": "LaBSE + Logistic",
    }
}

#-#####################################################################-#
# UTILITY FUNCTIONS ####
#-#####################################################################-#
def ensure_directory_exists(path: str) -> None:
    """Create directory if needed.
    Args:
        path: Directory path.
    """
    os.makedirs(path, exist_ok=True)


def get_training_datasets() -> list[str]:
    """List training datasets created in Stage 2."""

    files = glob.glob(os.path.join(TRAINING_DATASETS_DIR, "*.csv"))
    return sorted([f for f in files if "_training_" in os.path.basename(f)])


def configure_device() -> str:
    """Select CPU by default; optionally allow GPU."""
    if USE_GPU and torch.cuda.is_available():
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)  # allow CUDA
        torch.set_default_device("cuda")
        os.environ["SENTENCE_TRANSFORMERS_DEVICE"] = "cuda"
        return "cuda"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU
    torch.set_default_device("cpu")
    os.environ["SENTENCE_TRANSFORMERS_DEVICE"] = "cpu"
    return "cpu"


#-#####################################################################-#
# ASREVIEW SIMULATION ####
#-#####################################################################-#
def run_asreview_simulation(training_file: str, algorithm_name: str, algorithm_config: dict) -> tuple[bool, int, int, int]:
    """Execute one ASReview simulation and keep IDs/key studies.

    Args:
        training_file: Path to the training dataset.
        algorithm_name: Name of the algorithm configuration.
        algorithm_config: Dict with ASReview components.
    """
    dataset_name = os.path.basename(training_file)
    df = pd.read_csv(training_file)

    # Ensure text column exists and tally true positives
    if "title" not in df.columns:
        df["title"] = ""

    total_relevant = df["Included"].sum()

    # Select initial training set based on chosen mode
    training_records = (
        df[df["is_training_sample"] == True]
        if TRAINING_MODE == 1
        else df[df["sampling_order"] >= 0]
    )

    training_ids = training_records["position_before_simulation"].tolist()

    # Configure learner and run simulation
    learner = asr.ActiveLearningCycle(
        querier=algorithm_config["querier"],
        classifier=algorithm_config["classifier"],
        balancer=algorithm_config["balancer"],
        feature_extractor=algorithm_config["feature_extractor"],
    )

    sim = asr.Simulate(df, df["Included"], [learner])
    sim.label(training_ids)
    sim.review()

    # Collect screened and unscreened records
    results_df = sim._results.copy()
    df_complete = df.reset_index().copy()
    df_complete["record_id"] = df_complete.index

    screened_ids = set(results_df["record_id"].tolist())
    unscreened_records = df_complete[~df_complete["record_id"].isin(screened_ids)].copy()

    time_column = "time" if "time" in results_df.columns else "query_time"
    if not unscreened_records.empty:
        max_time = results_df[time_column].max() if time_column in results_df.columns else len(results_df)
        unscreened_data = {
            "record_id": unscreened_records["record_id"],
            "label": unscreened_records["Included"],
            "training_set": 0,
            time_column: range(int(max_time) + 1, int(max_time) + 1 + len(unscreened_records)),
        }
        for col in results_df.columns:
            if col not in unscreened_data:
                unscreened_data[col] = np.nan
        unscreened_addition = pd.DataFrame(unscreened_data)
        complete_simulation_df = pd.concat([results_df, unscreened_addition], ignore_index=True)
    else:
        complete_simulation_df = results_df.copy()

    complete_simulation_df = complete_simulation_df.sort_values(time_column).reset_index(drop=True)

    # Merge with original columns for ID/key-study preservation
    essential_orig_cols = ["record_id", "ID", "title", "is_training_sample", "is_key_study"]
    df_essential = df_complete[[col for col in essential_orig_cols if col in df_complete.columns]].copy()
    final_results = df_essential.merge(complete_simulation_df, on="record_id", how="left")
    final_results = final_results.rename(columns={"training_set": "screening_order"})

    # Determine status (training/screened/unscreened) and sequential order
    screened_rows = final_results[final_results["screening_order"].notna() & (final_results["screening_order"] != 0)]
    min_screened_time = screened_rows["time"].min() if not screened_rows.empty else float("inf")

    def determine_status(row):
        screening_order = row.get("screening_order")
        time_val = row.get("time")
        if pd.notna(screening_order) and screening_order != 0:
            return "screened"
        if (pd.isna(screening_order) or screening_order == 0) and pd.notna(time_val) and time_val <= min_screened_time:
            return "training"
        return "unscreened"

    final_results["status"] = final_results.apply(determine_status, axis=1)
    training_mask = final_results["status"] == "training"
    screened_mask = final_results["status"] == "screened"
    unscreened_mask = final_results["status"] == "unscreened"

    # Derive a unified sequential screening order across training/screened/unscreened
    final_results["sequential_screening_order"] = 0
    training_count = training_mask.sum()
    final_results.loc[training_mask, "sequential_screening_order"] = range(training_count)
    final_results.loc[screened_mask, "sequential_screening_order"] = final_results.loc[screened_mask, "screening_order"]
    max_screened = (
        final_results.loc[screened_mask, "screening_order"].max()
        if screened_mask.any()
        else training_count - 1
    )
    unscreened_count = unscreened_mask.sum()
    if unscreened_count > 0:
        start_unscreened = max_screened + 1
        final_results.loc[unscreened_mask, "sequential_screening_order"] = range(
            start_unscreened, start_unscreened + unscreened_count
        )

    if time_column in final_results.columns:
        final_results = final_results.sort_values(time_column).reset_index(drop=True)

    # Persist complete results for this dataset/algorithm
    base_name = dataset_name.replace(".csv", "")
    safe_base_name = base_name.replace("(", "_").replace(")", "_").replace(".", "_").replace(" ", "_")
    output_file = os.path.join(OUTPUT_DIR, f"{safe_base_name}_{algorithm_name}_complete.csv")
    ensure_directory_exists(os.path.dirname(output_file))
    final_results.to_csv(output_file, index=False)

    screening_length = len(final_results)
    relevant_found = (final_results["label"] == 1).sum()

    return True, screening_length, relevant_found, total_relevant


#-#####################################################################-#
# MAIN EXECUTION PIPELINE ####
#-#####################################################################-#
def main() -> int:
    """Run Stage 3 simulations."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = configure_device()

    # Ensure output targets exist
    ensure_directory_exists(OUTPUT_DIR)
    ensure_directory_exists(SUMMARY_DIR)

    # Collect inputs and algorithms to run
    training_files = get_training_datasets()
    algorithms_to_run = [SINGLE_ALGORITHM] if SINGLE_ALGORITHM else list(ALGORITHMS.keys())

    simulations_processed = 0
    skipped_existing = 0

    for training_file_path in training_files:
        dataset_name = os.path.basename(training_file_path)
        base_name = dataset_name.replace(".csv", "")
        safe_base_name = base_name.replace("(", "_").replace(")", "_").replace(".", "_").replace(" ", "_")
        for algorithm_name in algorithms_to_run:
            output_file = os.path.join(OUTPUT_DIR, f"{safe_base_name}_{algorithm_name}_complete.csv")
            if os.path.exists(output_file):
                skipped_existing += 1
                continue

            start_time = time.time()
            success, screening_length, relevant_found, total_relevant = run_asreview_simulation(
                training_file_path, algorithm_name, ALGORITHMS[algorithm_name]
            )
            simulation_time = time.time() - start_time

            simulations_processed += 1
            status = "SUCCESS" if success else "FAILED"
            print(
                f"{dataset_name} | {algorithm_name} ({device}) -> {status} | "
                f"{relevant_found}/{total_relevant} relevant | {screening_length} screened | {simulation_time:.1f}s"
            )

    print(f"Processed: {simulations_processed}, skipped existing: {skipped_existing}")
    return 0


if __name__ == "__main__":
    exit(main())
