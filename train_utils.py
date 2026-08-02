"""
Small, testable utility functions used by train.py.

Kept separate from train.py so they can be unit tested without
triggering the full training run (which train.py executes at
import/module-level).
"""

import os

FEATURE_COLUMNS = [
    'amount', 'time', 'amount_log', 'time_scaled',
    'amount_squared', 'high_amount_flag',
    'is_night', 'amount_bin', 'amount_ratio'
]


def resolve_dataset_path(base_dir: str) -> str:
    """
    Locate the processed training CSV, checking the script directory
    first and falling back to the current working directory.

    Raises:
        FileNotFoundError: if the dataset can't be found in either location.
    """
    candidate = os.path.join(base_dir, "data", "processed", "sar_dataset.csv")
    if os.path.exists(candidate):
        return candidate

    candidate = "data/processed/sar_dataset.csv"
    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(
        " Run preprocessing first to generate the dataset at data/processed/sar_dataset.csv"
    )
