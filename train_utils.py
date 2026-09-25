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


# Held-out split and decision threshold shared by train.py (at training
# time) and evaluate_model() (at display time), so the metrics shown in
# the app are computed on exactly the rows the model never trained on.
TEST_SIZE = 0.2
SPLIT_SEED = 42
DECISION_THRESHOLD = 0.9


def holdout_split(df):
    """Return (X_train, X_test, y_train, y_test) using the canonical split."""
    from sklearn.model_selection import train_test_split

    return train_test_split(
        df[FEATURE_COLUMNS], df['fraud'],
        test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=df['fraud'],
    )


def evaluate_model(model, df, threshold: float = DECISION_THRESHOLD) -> dict:
    """
    Score the canonical held-out split with ``model`` and return live
    metrics: accuracy, fraud-class precision/recall/F1, the 2x2 confusion
    matrix ([[TN, FP], [FN, TP]]) and the number of test rows.
    """
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    )

    _, X_test, _, y_test = holdout_split(df)
    y_pred = (model.predict_proba(X_test)[:, 1] > threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        "n_test": int(len(y_test)),
        "threshold": threshold,
    }
