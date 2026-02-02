# ASReview Simulation Pipeline

This project runs a multi-stage ASReview workflow: clean and filter datasets, generate training variants, run simulations, apply stopping rules, and visualize aggregated performance metrics. All stages are scripted for reproducibility with both CPU and CUDA-ready environments.

## Folder Tree
```
Simulation/
├── 01_scripts/
│   ├── 01_data_preparation.py
│   ├── 02_create_artificial_datasets.py
│   ├── 03_run_simulations.py
│   ├── 04_apply_stopping_rules.py
│   ├── 05_compute_summary_statistics.py
│   ├── 06_plot_stopping_rules.py
│   └── rename_results_files.py
├── 02_data/
│   ├── 00_meta_data/
│   ├── 01_input/
│   ├── 02_processed_input/
│   └── 03_screened_data/
├── 03_results/
└── requirements.txt
```

## Stage Overview
- **01_data_preparation.py**: Cleans CSVs in `02_data/01_input` (requires `abstract` and `Included`), filters out small/high-prevalence sets, adds `ID`, writes summary to `02_data/00_meta_data/stage1_data_preparation_summary.csv`.
- **02_create_artificial_datasets.py**: Samples each cleaned dataset to build training variants in `02_data/02_processed_input`; summary to `02_data/00_meta_data/stage2_training_datasets_summary.csv`.
- **03_run_simulations.py**: Runs ASReview on training sets from `02_data/02_processed_input`; outputs screened data to `02_data/03_screened_data` (files named `*_complete.csv`).
- **04_apply_stopping_rules.py**: Applies stopping rules to screened data; writes `03_results/stopping_rules_summary_key_parallel.csv`.
- **05_compute_summary_statistics.py**: Aggregates stopping-rule results into `03_results/stopping_rules_summary.csv`.
- **06_plot_stopping_rules.py**: Plots recall vs screening cost by prevalence band from `03_results/stopping_rules_summary.csv`.
- **rename_results_files.py**: Renames older Stage 3 outputs (`*_complete_results_*`) to the new `*_complete.csv` pattern.

## Key Outputs and Columns
### Stage 1 Summary (`02_data/00_meta_data/stage1_data_preparation_summary.csv`)
- `file_name`, `total_records`, `excluded_records`, `included_records`, `prevalence_percent`.

### Training Variants (Stage 2, `02_data/02_processed_input/*.csv`)
- `Included`, `abstract`, `ID`, `position_before_simulation`, `original_oac_dataset`, `training_dataset_id`, `is_training_sample`, `is_key_study`, `sampling_order`.

### Simulation Outputs (Stage 3, `02_data/03_screened_data/*_complete.csv`)
- `ID`, `title`/`abstract`, `is_training_sample`, `is_key_study`, `record_id`, `label`, `status` (`training`, `screened`, `unscreened`), `screening_order`, `sequential_screening_order`, `time` or `query_time`.

### Stopping-Rule Summary (Stage 4, `03_results/stopping_rules_summary_key_parallel.csv`)
- Per dataset/algorithm: `dataset_file`, `algorithm`, totals, training stats, selected band metrics, and per-rule (`A/B/C`) metrics such as screening cost, recall, keys found, breakouts.

### Aggregated Summary (Stage 5, `03_results/stopping_rules_summary.csv`)
- Per band/rule: mean/sd cost, mean/sd recall, proportions hitting recall targets, breakout usage, missing-key diagnostics, dataset counts, average training prevalence.

## Running the Pipeline
1. `python 01_scripts/01_data_preparation.py`
2. `python 01_scripts/02_create_artificial_datasets.py`
3. `python 01_scripts/03_run_simulations.py`
4. `python 01_scripts/04_apply_stopping_rules.py`
5. `python 01_scripts/05_compute_summary_statistics.py`
6. `python 01_scripts/06_plot_stopping_rules.py`

## Environments
- **CPU**: `pip install -r requirements.txt`
