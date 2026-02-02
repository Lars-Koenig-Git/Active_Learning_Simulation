#-#####################################################################-#
# STAGE 1: DATA PREPARATION FOR ASREVIEW SIMULATION PIPELINE ####
#-#####################################################################-#

import os
import glob
import shutil
from pathlib import Path
import pandas as pd

#-#####################################################################-#
# PROJECT CONFIGURATION ####
#-#####################################################################-#
PROJECT_PATH = str(Path(__file__).resolve().parent.parent)
INPUT_DIR = os.path.join(PROJECT_PATH, "02_data", "01_input")
SUMMARY_DIR = os.path.join(PROJECT_PATH, "02_data", "00_meta_data")


#-#####################################################################-#
# UTILITY FUNCTIONS ####
#-#####################################################################-#
def remove_spaces_from_filenames(input_dir: str) -> None:
    """Remove spaces from CSV filenames to prevent path issues.

    Args:
        input_dir: Directory containing CSV files to rename.
    """
    for path in glob.glob(os.path.join(input_dir, "*.csv")):
        dir_name, base = os.path.split(path)
        new_base = base.replace(" ", "")
        new_path = os.path.join(dir_name, new_base)
        if new_path != path:
            shutil.move(path, new_path)
            print(f"Renamed {base} -> {new_base}")


def normalize_and_validate_columns(df: pd.DataFrame, file_name: str) -> pd.DataFrame | None:
    """Ensure required columns exist and normalize casing for included/abstracts.

    Args:
        df: Input dataframe.
        file_name: Name of the CSV file (for logging).
    """
    cols_lower = {c.lower(): c for c in df.columns}

    # Map possible variants to expected names
    if "label_included" in cols_lower:
        df.rename(columns={cols_lower["label_included"]: "Included"}, inplace=True)
        cols_lower["included"] = "Included"

    if "included" not in cols_lower or "abstract" not in cols_lower:
        print(f"Skipped {file_name} - missing required columns 'abstract' and/or 'Included'")
        return None

    # Normalize casing
    df.rename(columns={cols_lower["included"]: "Included", cols_lower["abstract"]: "abstract"}, inplace=True)
    return df


def process_csv(file_path: str) -> bool:
    """Validate, filter, and rewrite a single dataset in place.

    Args:
        file_path: Path to the CSV file to process.
    """
    df = pd.read_csv(file_path)
    file_name = os.path.basename(file_path)

    df = normalize_and_validate_columns(df, file_name)
    if df is None:
        return False

    if len(df) <= 200:  # too few records to be useful
        os.remove(file_path)
        return False

    ones = (df["Included"] == 1).sum()
    zeros = (df["Included"] == 0).sum()
    total_labeled = ones + zeros
    if total_labeled == 0:
        print(f"Skipped {file_name} - Included column has no binary labels")
        return False

    prevalence = ones / total_labeled  # prevalence of included labels
    if prevalence > 0.5:
        os.remove(file_path)
        return False

    df["ID"] = range(1, len(df) + 1)
    df.to_csv(file_path, index=False)
    print(f"Processed {file_name} stored in place")
    return True


def generate_summary_statistics(csv_files: list[str]) -> pd.DataFrame:
    """Build simple prevalence/size stats for all processed datasets.

    Args:
        csv_files: List of processed CSV file paths.
    """
    rows = []
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        zeros = (df["Included"] == 0).sum()
        ones = (df["Included"] == 1).sum()
        prevalence_percent = ones / (ones + zeros) * 100 if (ones + zeros) > 0 else 0
        rows.append(
            {
                "file_name": os.path.basename(file_path),
                "total_records": len(df),
                "excluded_records": zeros,
                "included_records": ones,
                "prevalence_percent": prevalence_percent,
            }
        )
    return pd.DataFrame(rows)


#-#####################################################################-#
# MAIN EXECUTION PIPELINE ####
#-#####################################################################-#
def main() -> None:
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    remove_spaces_from_filenames(INPUT_DIR)

    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    processed_files: list[str] = []
    for path in csv_files:
        if process_csv(path):
            processed_files.append(path)

    if processed_files:
        summary = generate_summary_statistics(processed_files)
        summary_file = os.path.join(SUMMARY_DIR, "stage1_data_preparation_summary.csv")
        summary.to_csv(summary_file, index=False)
    else:
        print("No valid datasets found after processing.")


if __name__ == "__main__":
    main()
