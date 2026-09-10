"""
Tests that verify model.json and features.pkl are valid, loadable
artifacts consistent with what predict.py expects.
"""
import os
import joblib
import pytest
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def features():
    return joblib.load(os.path.join(BASE_DIR, "features.pkl"))


@pytest.fixture(scope="module")
def model():
    m = xgb.XGBClassifier()
    m.load_model(os.path.join(BASE_DIR, "model.json"))
    return m


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


def test_model_json_is_not_a_pickle_file():
    """Regression test: model.pkl (joblib/pickle) was migrated to
    model.json (XGBoost's native format) specifically because pickling
    the sklearn wrapper object isn't guaranteed version-portable across
    xgboost releases - it already emitted a compat UserWarning on load.
    Native format is explicitly designed to be portable. Confirm the
    committed artifact is genuinely JSON, not a pickle blob."""
    import json

    with open(os.path.join(BASE_DIR, "model.json")) as f:
        data = json.load(f)  # raises if this isn't valid JSON
    assert "learner" in data or "version" in data, (
        "model.json should have XGBoost's native model structure"
    )


def test_loading_model_json_emits_no_pickle_compat_warning():
    """Regression test: joblib.load(model.pkl) emitted
    'UserWarning: If you are loading a serialized model (like pickle
    in Python...)' on every single load. Loading the native format
    must not trigger this warning at all."""
    import warnings
    import xgboost as xgb

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = xgb.XGBClassifier()
        m.load_model(os.path.join(BASE_DIR, "model.json"))

    pickle_warnings = [w for w in caught if "pickle" in str(w.message).lower()]
    assert not pickle_warnings, (
        f"loading model.json should not emit pickle-compat warnings, got: {pickle_warnings}"
    )
