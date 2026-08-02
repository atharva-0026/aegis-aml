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
