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
    snippet = content[idx: idx + 600]
    assert "except Exception:" in snippet
    assert "pass" not in snippet.split("except Exception:")[1].split("\n")[1], (
        "load_processed_data's except block must not be a silent pass"
    )
    assert "logger." in snippet, "load_processed_data must log the failure"
