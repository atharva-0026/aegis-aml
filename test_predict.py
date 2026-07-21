"""
Tests for predict.py — transaction risk scoring.
"""
import pytest
from predict import predict_transaction


def test_output_shape():
    """Result should always contain the expected keys with correct types."""
    result = predict_transaction(1000, 3600)
    assert set(result.keys()) == {
        "fraud_probability", "prediction", "flagged_by_ml", "flagged_by_rules"
    }
    assert isinstance(result["fraud_probability"], float)
    assert result["prediction"] in ("Fraud", "Normal")
    assert isinstance(result["flagged_by_ml"], bool)
    assert isinstance(result["flagged_by_rules"], bool)


def test_probability_in_valid_range():
    result = predict_transaction(5000, 10000)
    assert 0.0 <= result["fraud_probability"] <= 1.0


def test_high_amount_triggers_rule_flag():
    """Amounts over 50,000 must always be rule-flagged regardless of ML score."""
    result = predict_transaction(60000, 40000)
    assert result["flagged_by_rules"] is True
    assert result["prediction"] == "Fraud"


def test_large_amount_early_triggers_rule_flag():
    """Amount > 30,000 combined with a small time offset should trigger the rule."""
    result = predict_transaction(35000, 1000)
    assert result["flagged_by_rules"] is True


def test_small_normal_transaction_not_rule_flagged():
    """A modest daytime transaction should not trip the hard-coded rules."""
    result = predict_transaction(500, 50000)
    assert result["flagged_by_rules"] is False


def test_zero_amount_does_not_crash():
    result = predict_transaction(0, 0)
    assert result["fraud_probability"] >= 0.0


def test_threshold_changes_ml_flag():
    """A very low threshold should be at least as sensitive as a high one."""
    low = predict_transaction(2000, 2000, threshold=0.0)
    high = predict_transaction(2000, 2000, threshold=0.999)
    assert low["flagged_by_ml"] is True
    assert high["flagged_by_ml"] is False
