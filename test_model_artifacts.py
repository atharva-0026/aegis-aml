"""
Tests that verify model.pkl and features.pkl are valid, loadable
artifacts consistent with what predict.py expects.
"""
import os
import joblib
import pytest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def features():
    return joblib.load(os.path.join(BASE_DIR, "features.pkl"))


@pytest.fixture(scope="module")
def model():
    return joblib.load(os.path.join(BASE_DIR, "model.pkl"))


def test_features_file_is_a_list(features):
    assert isinstance(features, list)
    assert len(features) > 0


def test_expected_feature_columns_present(features):
    expected = {
        "amount", "time", "amount_log", "time_scaled",
        "amount_squared", "high_amount_flag", "is_night",
        "amount_bin", "amount_ratio",
    }
    assert expected.issubset(set(features))


def test_model_has_predict_proba(model):
    assert hasattr(model, "predict_proba")


def test_model_is_binary_classifier(model, features):
    import pandas as pd
    sample = pd.DataFrame([{f: 0 for f in features}])
    probs = model.predict_proba(sample)
    assert probs.shape[1] == 2
