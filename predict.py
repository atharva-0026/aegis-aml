"""
Inference module for scoring transactions with the trained AML model.
"""

import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")
features_path = os.path.join(BASE_DIR, "features.pkl")

model = joblib.load(model_path)
features = joblib.load(features_path)


def predict_transaction(amount, time, threshold=0.6):
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
    if amount < 0:
        raise ValueError(f"amount must be non-negative, got {amount}")
    if time < 0:
        raise ValueError(f"time must be non-negative, got {time}")

    # Create features for inference
    data = pd.DataFrame([{
        'amount': amount,
        'time': time
    }])

    data['amount_log'] = np.log1p(data['amount'])
    data['time_scaled'] = data['time'] / 172800

    data['amount_squared'] = data['amount'] ** 2
    data['high_amount_flag'] = (data['amount'] > 10000).astype(int)

    data['is_night'] = (data['time'] % 86400 < 21600).astype(int)
    data['amount_bin'] = pd.cut(data['amount'], bins=[0, 100, 1000, 10000, 100000, np.inf], labels=False)
    data['amount_ratio'] = data['amount'] / 5000 

    data.fillna(0, inplace=True)

    # Ensure all features align with training features
    for col in features:
        if col not in data.columns:
            data[col] = 0
            
    data = data[features]

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



if __name__ == "__main__":
    result = predict_transaction(80000, 1000)
    print(result)