"""
Small, testable utility functions used by train.py.

Kept separate from train.py so they can be unit tested without
triggering the full training run (which train.py executes at
import/module-level).
"""

import os

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

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


# Must match train.py's split exactly (test_size, random_state, stratify)
# so this reproduces the same held-out rows train.py evaluated against,
# and the classification threshold train.py reports its metrics at.
TEST_SIZE = 0.2
RANDOM_STATE = 42
CLASSIFICATION_THRESHOLD = 0.9


def evaluate_model(base_dir: str, model) -> dict:
    """
    Score `model` against the same held-out split train.py produces,
    so callers (e.g. app.py's metrics panel) can show real, current
    accuracy/F1/confusion-matrix numbers instead of frozen ones.

    Args:
        base_dir: directory to resolve the processed dataset from.
        model: a fitted classifier exposing predict_proba(), scored on
            FEATURE_COLUMNS.

    Returns:
        {"accuracy": float, "f1_score": float,
         "confusion_matrix": [[tn, fp], [fn, tp]]}
    """
    df = pd.read_csv(resolve_dataset_path(base_dir))
    X = df[FEATURE_COLUMNS]
    y = df['fraud']

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs > CLASSIFICATION_THRESHOLD).astype(int)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
    }
