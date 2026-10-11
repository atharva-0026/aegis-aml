import pandas as pd
import pytest

from batch_utils import DECISION_COL, PROBABILITY_COL, flagged_only, score_batch, summarize_batch


def _fake_predict(amount, time, threshold=0.5):
    if amount < 0:
        raise ValueError("amount must be non-negative")
    prob = 0.95 if amount >= 50000 else 0.1
    return {"fraud_probability": prob, "prediction": "Fraud" if prob >= threshold else "Normal"}


@pytest.fixture
def batch():
    return pd.DataFrame({
        "amount": [12000.0, 80000.0, -5.0, 55000.0],
        "time": [45000, 1000, 10, 80000],
    })


def test_score_batch_adds_columns_and_defaults(batch):
    scored = score_batch(batch, _fake_predict, threshold=0.5)
    assert list(scored[DECISION_COL]) == ["Normal", "Fraud", "Error: amount must be non-negative", "Fraud"]
    assert scored[PROBABILITY_COL].iloc[2] is None or pd.isna(scored[PROBABILITY_COL].iloc[2])
    assert (scored["location"] == "India").all()
    assert (scored["transaction_type"] == "payment").all()
    assert DECISION_COL not in batch.columns, "must not mutate the uploaded frame"


def test_score_batch_respects_threshold(batch):
    scored = score_batch(batch, _fake_predict, threshold=0.99)
    assert "Fraud" not in set(scored[DECISION_COL])


def test_summarize_batch(batch):
    summary = summarize_batch(score_batch(batch, _fake_predict, threshold=0.5))
    assert summary == {
        "total": 4,
        "flagged": 2,
        "errors": 1,
        "flag_rate": pytest.approx(2 / 3),
        "flagged_volume": 135000.0,
    }


def test_summarize_batch_all_errors_does_not_divide_by_zero():
    df = pd.DataFrame({"amount": [-1.0], "time": [0]})
    summary = summarize_batch(score_batch(df, _fake_predict, threshold=0.5))
    assert summary["flag_rate"] == 0.0 and summary["errors"] == 1


def test_flagged_only(batch):
    out = flagged_only(score_batch(batch, _fake_predict, threshold=0.5))
    assert list(out["amount"]) == [80000.0, 55000.0]
