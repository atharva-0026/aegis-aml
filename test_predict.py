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


def test_negative_amount_raises():
    with pytest.raises(ValueError):
        predict_transaction(-100, 1000)


def test_negative_time_raises():
    with pytest.raises(ValueError):
        predict_transaction(100, -1)


def test_explain_prediction_returns_top_factors():
    from predict import explain_prediction

    result = explain_prediction(60000, 1000, top_n=3)
    assert "top_factors" in result
    assert len(result["top_factors"]) == 3
    for factor in result["top_factors"]:
        assert "feature" in factor
        assert "shap_value" in factor
        assert isinstance(factor["shap_value"], float)


def test_explain_prediction_sorted_by_absolute_impact():
    from predict import explain_prediction

    result = explain_prediction(60000, 1000, top_n=5)
    abs_values = [abs(f["shap_value"]) for f in result["top_factors"]]
    assert abs_values == sorted(abs_values, reverse=True)


def test_explain_prediction_respects_top_n():
    from predict import explain_prediction

    result = explain_prediction(1000, 5000, top_n=2)
    assert len(result["top_factors"]) == 2


def test_explain_prediction_raises_on_negative_amount():
    from predict import explain_prediction

    with pytest.raises(ValueError):
        explain_prediction(-100, 1000)


def test_amount_bin_matches_training_edges():
    """Regression test: amount_bin must use the same bin edges as
    preprocessing.py's training-time computation, not arbitrary fixed bins."""
    import pandas as pd
    from predict import TRAIN_AMOUNT_BIN_EDGES

    sample = pd.DataFrame({"amount": [500, 2500, 5000, 9000]})
    result = pd.cut(
        sample["amount"], bins=TRAIN_AMOUNT_BIN_EDGES, labels=False, include_lowest=True
    )
    assert list(result) == [0, 1, 2, 4]


def test_amount_ratio_uses_training_mean():
    from predict import TRAIN_AMOUNT_MEAN

    assert abs(TRAIN_AMOUNT_MEAN - 87.51) < 0.01
