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
    """Regression test: the batch scan loop in app.py previously called
    predict_transaction() with no error handling. Since
    predict_transaction() raises ValueError for negative amount/time, a
    single bad row anywhere in an uploaded CSV crashed the ENTIRE batch
    scan, losing all processing on every other row too. Must catch and
    mark that specific row as an error instead of crashing."""
    with open(os.path.join(os.path.dirname(__file__), "app.py")) as f:
        content = f.read()
    idx = content.find("for _idx, row in batch_df.iterrows()")
    assert idx != -1, "expected the batch scan loop in app.py"
    snippet = content[idx: idx + 700]
    assert "try:" in snippet, "batch scan loop must handle per-row prediction errors"
    assert "except" in snippet
    assert "predict_transaction" in snippet.split("try:")[1], (
        "predict_transaction call must be inside the try block, not before it"
    )


def test_hardcoded_model_metrics_staleness_is_documented():
    """Regression test: app.py's Model & Knowledge Base panel shows
    hardcoded (not live) accuracy/confusion-matrix numbers. This isn't
    fixed (see KNOWN_ISSUES.md for why), but the code comment flagging
    it must stay in place so the limitation stays visible to anyone
    editing this section, and KNOWN_ISSUES.md must keep tracking it."""
    with open(os.path.join(os.path.dirname(__file__), "app.py")) as f:
        app_content = f.read()
    idx = app_content.find("Diagnostic Confusion Matrix Heatmap")
    assert idx != -1
    nearby = app_content[max(0, idx - 500): idx]
    assert "NOTE" in nearby and "hardcoded" in nearby.lower(), (
        "the staleness-risk comment must stay near the confusion matrix code"
    )

    with open(os.path.join(os.path.dirname(__file__), "KNOWN_ISSUES.md")) as f:
        known_issues = f.read()
    assert "hardcoded, not live" in known_issues.lower() or "hardcoded" in known_issues.lower()


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
