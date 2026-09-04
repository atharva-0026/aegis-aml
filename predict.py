"""
Inference module for scoring transactions with the trained AML model.
"""

from __future__ import annotations

import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")
features_path = os.path.join(BASE_DIR, "features.pkl")

model = joblib.load(model_path)
features = joblib.load(features_path)

# Exact bin edges and mean derived from data/processed/sar_dataset.csv
# (the actual training data), replacing the previously mismatched fixed
# bins and hardcoded divisor. See KNOWN_ISSUES.md for background.
TRAIN_AMOUNT_BIN_EDGES = [-10.0, 2000.0, 4000.0, 6000.0, 8000.0, 10000.0]
TRAIN_AMOUNT_MEAN = 87.5126602

# Lazily constructed — SHAP's TreeExplainer setup has a small fixed cost,
# no need to pay it on import for callers who never use explain_prediction.
_explainer = None


def _build_features(amount: float, time: float) -> pd.DataFrame:
    """
    Build the model-ready feature row for a single transaction.

    Raises:
        ValueError: if amount or time is negative.
    """
    if amount < 0:
        raise ValueError(f"amount must be non-negative, got {amount}")
    if time < 0:
        raise ValueError(f"time must be non-negative, got {time}")

    data = pd.DataFrame([{
        'amount': amount,
        'time': time
    }])

    data['amount_log'] = np.log1p(data['amount'])
    data['time_scaled'] = data['time'] / 172800

    data['amount_squared'] = data['amount'] ** 2
    data['high_amount_flag'] = (data['amount'] > 10000).astype(int)

    data['is_night'] = (data['time'] % 86400 < 21600).astype(int)
    # Matches preprocessing.py's training-time computation exactly via
    # TRAIN_AMOUNT_BIN_EDGES/TRAIN_AMOUNT_MEAN. See KNOWN_ISSUES.md.
    data['amount_bin'] = pd.cut(
        data['amount'], bins=TRAIN_AMOUNT_BIN_EDGES, labels=False, include_lowest=True
    )
    data['amount_ratio'] = data['amount'] / TRAIN_AMOUNT_MEAN

    data.fillna(0, inplace=True)

    # Ensure all features align with training features
    for col in features:
        if col not in data.columns:
            data[col] = 0

    return data[features]


def predict_transaction(amount: float, time: float, threshold: float = 0.6) -> dict:
    """
    Score a single transaction for AML risk.

    Args:
        amount: transaction amount
        time: transaction time offset in seconds
        threshold: probability cutoff for flagging as suspicious

    Returns:
        Risk prediction with probability score.

    Raises:
        ValueError: if amount or time is negative.
    """
    data = _build_features(amount, time)

    # Predict probability
    prob = float(model.predict_proba(data)[0][1])

    # Rule-based heuristics
    rule_flag = (
        amount > 50000 or
        (amount > 30000 and time < 5000)
    )

    ml_flag = prob > threshold
    flagged = ml_flag or rule_flag

    return {
        "fraud_probability": round(prob, 4),
        "prediction": "Fraud" if flagged else "Normal",
        "flagged_by_ml": bool(ml_flag),
        "flagged_by_rules": bool(rule_flag)
    }


def explain_prediction(amount: float, time: float, top_n: int = 3) -> dict:
    """
    Explain a single transaction's ML risk score using SHAP.

    Unlike the fixed rule-based flags in predict_transaction (hardcoded
    amount thresholds), this shows which model features actually pushed
    this specific prediction toward or away from "Fraud", using
    TreeExplainer on the trained XGBoost model.

    Args:
        amount: transaction amount
        time: transaction time offset in seconds
        top_n: how many top contributing features to return

    Returns:
        {"top_factors": [{"feature": str, "shap_value": float}, ...]}
        sorted by absolute contribution, most impactful first. Positive
        shap_value pushes toward "Fraud", negative pushes toward "Normal".

    Raises:
        ValueError: if amount or time is negative.
        ImportError: if the optional `shap` dependency isn't installed.
    """
    global _explainer

    try:
        import shap
    except ImportError as e:
        raise ImportError(
            "explain_prediction requires the optional 'shap' dependency. "
            "Install with: pip install shap"
        ) from e

    data = _build_features(amount, time)

    if _explainer is None:
        _explainer = shap.TreeExplainer(model)

    shap_values = _explainer.shap_values(data)[0]

    contributions = sorted(
        zip(features, shap_values, strict=True),
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )[:top_n]

    return {
        "top_factors": [
            {"feature": name, "shap_value": round(float(value), 4)}
            for name, value in contributions
        ]
    }


if __name__ == "__main__":
    result = predict_transaction(80000, 1000)
    print(result)