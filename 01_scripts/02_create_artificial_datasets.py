#!/usr/bin/env python3

#-#####################################################################-#
# STAGE 2: CREATE TRAINING DATASETS FOR ASREVIEW SIMULATION PIPELINE ####
#-#####################################################################-#

import glob
import os
import random

import numpy as np
import pandas as pd
from pathlib import Path

#-#####################################################################-#
# PROJECT CONFIGURATION ####
#-#####################################################################-#
PROJECT_PATH = str(Path(__file__).resolve().parent.parent)
INPUT_DIR = os.path.join(PROJECT_PATH, "02_data", "01_input")
OUTPUT_DIR = os.path.join(PROJECT_PATH, "02_data", "02_processed_input")
SUMMARY_DIR = os.path.join(PROJECT_PATH, "02_data", "00_meta_data")

# Sampling parameters
INITIAL_SAMPLE_SIZE = 100
ADDITIONAL_SAMPLE_SIZE = 1
NUM_TRAINING_VARIANTS = 100
RANDOM_SEED_BASE = 42


#-#####################################################################-#
# UTILITY FUNCTIONS ####
#-#####################################################################-#
def ensure_directory_exists(path: str) -> None:
    """Create directory if it doesn't exist.

    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def get_input_datasets(input_dir: str) -> list[str]:
    """Return sorted CSV inputs.

    Args:
        input_dir: Directory containing cleaned input datasets.
    """
    return sorted(glob.glob(os.path.join(input_dir, "*.csv")))


def generate_unique_random_seeds(total_needed: int) -> list[int]:
    """Generate unique random seeds from 1..10000.

    Args:
        total_needed: Number of seeds to generate.
    """
    random.seed(RANDOM_SEED_BASE)
    return random.sample(range(1, 10001), total_needed)


#-#####################################################################-#
# CORE SAMPLING ALGORITHM ####
#-#####################################################################-#
def sample_until_relevant_found(
    df: pd.DataFrame,
    initial_size: int = INITIAL_SAMPLE_SIZE,
    additional_size: int = ADDITIONAL_SAMPLE_SIZE,
    random_seed: int = RANDOM_SEED_BASE,
) -> dict | None:
    """Sample rows until at least one relevant (Included=1) record is present.

    Args:
        df: Input dataframe.
        initial_size: Initial sample size.
        additional_size: Size of each additional sample if none relevant yet.
        random_seed: Seed to drive reproducible sampling.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    sampled_indices: list[int] = []
    available_indices = list(range(len(df)))

    initial_sample = random.sample(available_indices, min(initial_size, len(available_indices)))
    sampled_indices.extend(initial_sample)

    sampled_data = df.iloc[sampled_indices]
    relevant_indices = sampled_data[sampled_data["Included"] == 1].index.tolist()
    iteration = 1

    while len(relevant_indices) == 0:
        remaining_indices = [i for i in available_indices if i not in sampled_indices]
        if not remaining_indices:
            break
        additional_sample = random.sample(remaining_indices, min(additional_size, len(remaining_indices)))
        sampled_indices.extend(additional_sample)

        sampled_data = df.iloc[sampled_indices]
        relevant_indices = sampled_data[sampled_data["Included"] == 1].index.tolist()
        iteration += 1
        if iteration > 1000:
            break

    if len(relevant_indices) == 0:
        return None

    num_sampled_relevant = len(relevant_indices)
    num_sampled_irrelevant = len(sampled_indices) - num_sampled_relevant
    sampling_percentage = (
        num_sampled_relevant / len(sampled_indices) * 100 if sampled_indices else 0
    )

    training_relevant_idx = random.choice(relevant_indices)
    key_study_indices = [idx for idx in relevant_indices if idx != training_relevant_idx]

    return {
        "sampled_indices": sampled_indices,
        "training_relevant_idx": training_relevant_idx,
        "key_study_indices": key_study_indices,
        "total_relevant_found": len(relevant_indices),
        "random_seed_used": random_seed,
        "sampling_stats": {
            "total_sampled": len(sampled_indices),
            "sampled_relevant": num_sampled_relevant,
            "sampled_irrelevant": num_sampled_irrelevant,
            "sampling_percentage": sampling_percentage,
            "iterations_needed": iteration,
        },
    }


#-#####################################################################-#
# TRAINING DATASET CREATION ####
#-#####################################################################-#
def create_training_dataset(
    df: pd.DataFrame, sampling_result: dict, variant_id: int, original_filename: str
) -> pd.DataFrame:
    """Apply ASReview markings for one training variant.

    Args:
        df: Original dataframe.
        sampling_result: Sampling metadata from sample_until_relevant_found.
        variant_id: Variant number (1-based).
        original_filename: Name of the source dataset file.
    """
    modified_df = df.copy()
    modified_df["position_before_simulation"] = range(len(df))
    modified_df["original_oac_dataset"] = original_filename
    modified_df["training_dataset_id"] = variant_id
    modified_df["is_training_sample"] = False
    modified_df["is_key_study"] = False
    modified_df["sampling_order"] = -1

    for order, idx in enumerate(sampling_result["sampled_indices"]):
        modified_df.loc[idx, "sampling_order"] = order

    training_sample_indices = [
        idx for idx in sampling_result["sampled_indices"] if idx not in sampling_result["key_study_indices"]
    ]
    for idx in training_sample_indices:
        modified_df.loc[idx, "is_training_sample"] = True
    for idx in sampling_result["key_study_indices"]:
        modified_df.loc[idx, "is_key_study"] = True

    return modified_df


#-#####################################################################-#
# DATASET PROCESSING ####
#-#####################################################################-#
def process_single_dataset(
    dataset_file_path: str, dataset_index: int, variant_seeds: list[int]
) -> tuple[list[dict], list[dict]]:
    """Create all training variants for a single dataset; return stats.

    Args:
        dataset_file_path: Path to the dataset CSV.
        dataset_index: Index of the dataset in processing order.
        variant_seeds: List of seeds, one per variant.
    """
    dataset_name = os.path.basename(dataset_file_path)

    df = pd.read_csv(dataset_file_path)
    total_relevant = df["Included"].sum()

    total_records = len(df)
    prevalence = total_relevant / total_records * 100

    summary_stats: list[dict] = []
    sampling_stats: list[dict] = []

    for variant in range(NUM_TRAINING_VARIANTS):
        variant_seed = variant_seeds[variant]
        sampling_result = sample_until_relevant_found(
            df,
            initial_size=INITIAL_SAMPLE_SIZE,
            additional_size=ADDITIONAL_SAMPLE_SIZE,
            random_seed=variant_seed,
        )
        if sampling_result is None:
            continue

        training_dataset = create_training_dataset(df, sampling_result, variant + 1, dataset_name)
        base_name = dataset_name.replace(".csv", "")
        output_filename = f"{base_name}_training_{variant + 1:02d}_seed{variant_seed}.csv"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        training_dataset.to_csv(output_path, index=False)

        stats = sampling_result["sampling_stats"]
        summary_stats.append(
            {
                "original_dataset": dataset_name,
                "dataset_index": dataset_index,
                "training_variant": variant + 1,
                "random_seed": variant_seed,
                "total_records": total_records,
                "total_relevant": total_relevant,
                "prevalence_percent": prevalence,
                "total_sampled": stats["total_sampled"],
                "sampled_relevant": stats["sampled_relevant"],
                "sampled_irrelevant": stats["sampled_irrelevant"],
                "sampling_percentage": stats["sampling_percentage"],
                "iterations_needed": stats["iterations_needed"],
                "key_study_count": len(sampling_result["key_study_indices"]),
                "output_filename": output_filename,
            }
        )

        sampling_stats.append(
            {
                "original_dataset": dataset_name,
                "training_variant": variant + 1,
                "random_seed": variant_seed,
                "total_sampled": stats["total_sampled"],
                "sampled_relevant": stats["sampled_relevant"],
                "iterations_needed": stats["iterations_needed"],
                "sampling_percentage": stats["sampling_percentage"],
                "output_filename": output_filename,
            }
        )

    return summary_stats, sampling_stats


#-#####################################################################-#
# MAIN EXECUTION PIPELINE ####
#-#####################################################################-#
def main() -> int:
    """Run Stage 2: generate training datasets and summary."""
    ensure_directory_exists(OUTPUT_DIR)
    ensure_directory_exists(SUMMARY_DIR)

    input_files = get_input_datasets(INPUT_DIR)
    if not input_files:
        print("No input datasets found in 02_data/01_input. Nothing to do.")
        return 0

    total_seeds_needed = len(input_files) * NUM_TRAINING_VARIANTS
    unique_seeds = generate_unique_random_seeds(total_seeds_needed)

    all_summary_stats: list[dict] = []
    seed_index = 0

    for dataset_index, dataset_file in enumerate(input_files):
        variant_seeds = unique_seeds[seed_index : seed_index + NUM_TRAINING_VARIANTS]
        seed_index += NUM_TRAINING_VARIANTS

        summary_stats, _ = process_single_dataset(dataset_file, dataset_index, variant_seeds)
        all_summary_stats.extend(summary_stats)
        if summary_stats:
            print(f"Processed {dataset_file} -> {len(summary_stats)} training variant(s)")

    if all_summary_stats:
        comprehensive_summary = pd.DataFrame(all_summary_stats)
        summary_path = os.path.join(SUMMARY_DIR, "stage2_training_datasets_summary.csv")
        comprehensive_summary.to_csv(summary_path, index=False)
        print(f"Wrote summary to {summary_path}")
    else:
        print("No training datasets were created.")
    return 0


if __name__ == "__main__":
    exit(main())
