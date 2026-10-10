"""
Batch scoring helpers for the Batch Scan Centre (app.py, Module 4).

Kept out of app.py so the per-row error handling and the summary numbers
shown above the ledger can be unit-tested without a Streamlit runtime.
"""
import pandas as pd

DECISION_COL = "Decision Audit"
PROBABILITY_COL = "Anomaly Probability"


def score_batch(batch_df: pd.DataFrame, predict_fn, threshold: float) -> pd.DataFrame:
    """
    Returns a copy of batch_df with PROBABILITY_COL and DECISION_COL added.

    predict_fn(amount, time, threshold=...) must return a dict with
    "fraud_probability" and "prediction" (predict.predict_transaction).
    A row that raises ValueError/TypeError (negative amount, a non-numeric
    cell, ...) gets probability None and an "Error: ..." decision instead
    of aborting the whole batch.
    """
    out = batch_df.copy()
    if "location" not in out.columns:
        out["location"] = "India"
    if "transaction_type" not in out.columns:
        out["transaction_type"] = "payment"

    scores, decisions = [], []
    for _idx, row in out.iterrows():
        try:
            res = predict_fn(row["amount"], row["time"], threshold=threshold)
            scores.append(res["fraud_probability"])
            decisions.append(res["prediction"])
        except (ValueError, TypeError) as e:
            scores.append(None)
            decisions.append(f"Error: {e}")

    out[PROBABILITY_COL] = scores
    out[DECISION_COL] = decisions
    return out


def summarize_batch(scored_df: pd.DataFrame) -> dict:
    """Headline numbers for a scored batch: row/flag/error counts and flagged volume."""
    decisions = scored_df[DECISION_COL].astype(str)
    flagged = decisions == "Fraud"
    errors = decisions.str.startswith("Error:")
    amounts = pd.to_numeric(scored_df.loc[flagged, "amount"], errors="coerce").fillna(0)
    total = len(scored_df)
    scored = total - int(errors.sum())
    return {
        "total": total,
        "flagged": int(flagged.sum()),
        "errors": int(errors.sum()),
        "flag_rate": (int(flagged.sum()) / scored) if scored else 0.0,
        "flagged_volume": float(amounts.sum()),
    }


def flagged_only(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Rows the model or rules flagged as Fraud."""
    return scored_df[scored_df[DECISION_COL] == "Fraud"]
