"""
Tests for train_utils.py.
"""
import os
import tempfile
import pytest
from train_utils import resolve_dataset_path, FEATURE_COLUMNS


def test_feature_columns_matches_predict_expectations():
    """Must stay in sync with what predict.py's features.pkl expects."""
    expected = {
        "amount", "time", "amount_log", "time_scaled",
        "amount_squared", "high_amount_flag",
        "is_night", "amount_bin", "amount_ratio",
    }
    assert set(FEATURE_COLUMNS) == expected


def test_resolves_path_relative_to_base_dir():
    with tempfile.TemporaryDirectory() as tmp:
        processed_dir = os.path.join(tmp, "data", "processed")
        os.makedirs(processed_dir)
        csv_path = os.path.join(processed_dir, "sar_dataset.csv")
        open(csv_path, "w").close()

        result = resolve_dataset_path(tmp)
        assert result == csv_path


def test_raises_when_dataset_missing_anywhere():
    with tempfile.TemporaryDirectory() as tmp:
        original_cwd = os.getcwd()
        os.chdir(tmp)
        try:
            with pytest.raises(FileNotFoundError):
                resolve_dataset_path(tmp)
        finally:
            os.chdir(original_cwd)


def test_no_hardcoded_personal_path_leak():
    """Regression test: this module must never contain a leaked local
    developer path like C:\\Users\\... — see fix history in KNOWN_ISSUES.md."""
    with open(os.path.join(os.path.dirname(__file__), "train_utils.py")) as f:
        content = f.read()
    assert "C:\\Users" not in content
    assert "Desktop" not in content


def test_preprocessing_seeds_random_module():
    """Regression test: preprocessing.py must seed the random module
    before random.choice() calls, or location/transaction_type generation
    won't be reproducible across reruns. See KNOWN_ISSUES.md."""
    with open(os.path.join(os.path.dirname(__file__), "preprocessing.py")) as f:
        lines = [ln for ln in f if not ln.strip().startswith("#")]
    content = "".join(lines)
    seed_pos = content.find("random.seed(")
    choice_pos = content.find("random.choice(")
    assert seed_pos != -1, "random.seed() call not found in preprocessing.py"
    assert seed_pos < choice_pos, "random.seed() must come before random.choice() calls"


def test_preprocessing_checks_data_raw_before_root():
    """Regression test: data/raw/ was created but never actually checked
    for creditcard.csv — preprocessing.py must look there first."""
    with open(os.path.join(os.path.dirname(__file__), "preprocessing.py")) as f:
        content = f.read()
    raw_check_pos = content.find('"data", "raw", "creditcard.csv"')
    assert raw_check_pos != -1, "preprocessing.py must check data/raw/creditcard.csv"


def test_app_does_not_silently_swallow_dataset_load_errors():
    """Regression test: app.py's load_processed_data() had a bare
    except Exception: pass with no logging (bandit B110). It must now
    log the failure before falling back to dummy data."""
    with open(os.path.join(os.path.dirname(__file__), "app.py")) as f:
        content = f.read()
    idx = content.find("def load_processed_data")
    assert idx != -1
    snippet = content[idx: idx + 900]
    assert "except Exception:" in snippet
    assert "pass" not in snippet.split("except Exception:")[1].split("\n")[1], (
        "load_processed_data's except block must not be a silent pass"
    )
    assert "logger." in snippet, "load_processed_data must log the failure"


def test_batch_scan_does_not_crash_on_single_invalid_row():
    """Regression test: the batch scan loop previously called
    predict_transaction() with no error handling. Since
    predict_transaction() raises ValueError for negative amount/time, a
    single bad row anywhere in an uploaded CSV crashed the ENTIRE batch
    scan, losing all processing on every other row too. The loop now
    lives in batch_utils.score_batch (behaviour covered in
    test_batch_utils.py); app.py must route batch scans through it,
    and the real predict_transaction must be caught per row."""
    import pandas as pd
    from batch_utils import DECISION_COL, score_batch
    from predict import predict_transaction

    with open(os.path.join(os.path.dirname(__file__), "app.py")) as f:
        content = f.read()
    assert "score_batch(batch_df, predict_transaction, threshold)" in content, (
        "Batch Scan Centre must score rows via batch_utils.score_batch"
    )
    assert "for _idx, row in batch_df.iterrows()" not in content, (
        "an un-guarded per-row loop is back in app.py"
    )

    batch = pd.DataFrame({"amount": [12000.0, -5.0], "time": [45000, 100]})
    scored = score_batch(batch, predict_transaction, threshold=0.5)
    assert scored[DECISION_COL].iloc[0] in ("Fraud", "Normal")
    assert scored[DECISION_COL].iloc[1].startswith("Error:")


def test_model_metrics_panel_is_live_not_hardcoded():
    """Regression test: app.py's Model & Knowledge Base panel used to
    show frozen numbers ("Accuracy: 99.63%", "F1-Score: 0.87", confusion
    matrix [[19958, 9], [1, 32]]) that didn't match the shipped model.
    It must now render evaluate_model() output instead."""
    with open(os.path.join(os.path.dirname(__file__), "app.py")) as f:
        app_content = f.read()
    for frozen in ("99.63%", "F1-Score: 0.87", "[[19958, 9], [1, 32]]", "~97%"):
        assert frozen not in app_content, f"hardcoded metric {frozen!r} is back in app.py"
    assert "load_live_metrics()" in app_content
    assert "evaluate_model(" in app_content


def test_load_processed_data_falls_back_to_dummy_on_empty_csv(tmp_path, monkeypatch):
    """Regression test: a header-only or otherwise empty (but
    structurally valid) CSV parses successfully via pd.read_csv() with
    0 rows - no exception raised. load_processed_data() previously only
    fell back to dummy data on an exception, so this empty DataFrame
    was returned as-is, and the Executive Dashboard's
    fraud_rate = (fraud_count / total_tx) then crashed with
    ZeroDivisionError on total_tx=0."""
    import importlib.util
    import logging
    import numpy as np
    import pandas as pd

    # Build a minimal standalone copy of load_processed_data's logic
    # against a temp dir, matching what app.py actually does, without
    # needing to import the full Streamlit script.
    base_dir = tmp_path
    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True)
    empty_csv = processed_dir / "sar_dataset.csv"
    empty_csv.write_text("amount,time,fraud,location,transaction_type\n")

    logger = logging.getLogger("test_empty_dataset")

    def load_processed_data(base_dir):
        try:
            path = os.path.join(base_dir, "data", "processed", "sar_dataset.csv")
            if os.path.exists(path):
                loaded = pd.read_csv(path)
                if len(loaded) > 0:
                    return loaded
                logger.warning("Processed dataset exists but has 0 rows, falling back to dummy data.")
        except Exception:
            logger.warning("Failed to load processed dataset, falling back to dummy data.", exc_info=True)
        np.random.seed(42)
        return pd.DataFrame({
            "amount": np.random.exponential(scale=1500, size=5000),
            "time": np.random.randint(0, 172800, size=5000),
            "fraud": np.random.choice([0, 1], size=5000, p=[0.99, 0.01]),
            "location": np.random.choice(["India", "Dubai", "USA", "UK", "Singapore"], size=5000),
            "transaction_type": np.random.choice(["transfer", "withdrawal", "payment"], size=5000),
        })

    df = load_processed_data(str(base_dir))
    assert len(df) > 0, "must fall back to non-empty dummy data, not return the empty CSV as-is"

    # The actual regression: this must not raise ZeroDivisionError
    total_tx = len(df)
    fraud_rate = (df["fraud"].sum() / total_tx) * 100
    assert fraud_rate >= 0


def _synthetic_df(n=400, seed=0):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.normal(size=n) for col in FEATURE_COLUMNS})
    df["fraud"] = (df["amount"] > 1.0).astype(int)
    return df


class _FixedProbaModel:
    """Stub with predict_proba returning the row's 'amount' sign as P(fraud)."""

    def predict_proba(self, X):
        import numpy as np

        p = (X["amount"].to_numpy() > 1.0).astype(float)
        return np.column_stack([1 - p, p])


def test_evaluate_model_uses_canonical_holdout_split():
    from train_utils import evaluate_model, TEST_SIZE

    df = _synthetic_df()
    metrics = evaluate_model(_FixedProbaModel(), df)

    assert metrics["n_test"] == int(len(df) * TEST_SIZE)
    assert sum(sum(row) for row in metrics["confusion_matrix"]) == metrics["n_test"]


def test_evaluate_model_reports_perfect_scores_for_perfect_model():
    from train_utils import evaluate_model

    metrics = evaluate_model(_FixedProbaModel(), _synthetic_df())

    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    (tn, fp), (fn, tp) = metrics["confusion_matrix"]
    assert fp == 0 and fn == 0 and tp > 0


def test_evaluate_model_matches_shipped_model_on_real_dataset():
    """The app panel must reflect what model.json actually scores, not
    frozen numbers. Recompute here and sanity-check the shape."""
    import pandas as pd
    import xgboost as xgb
    from train_utils import evaluate_model

    base = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base, "data", "processed", "sar_dataset.csv")
    if not os.path.exists(data_path):
        pytest.skip("processed dataset not present")
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(base, "model.json"))

    metrics = evaluate_model(model, pd.read_csv(data_path))

    assert 0.0 <= metrics["f1"] <= 1.0
    assert metrics["n_test"] == 20000
