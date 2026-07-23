"""
Tests for nav_utils.clean_nav.
"""
from nav_utils import clean_nav


def test_strips_emoji_prefix():
    assert clean_nav(" Single Scan") == "Single Scan"


def test_plain_ascii_unchanged():
    assert clean_nav("Executive Dashboard") == "Executive Dashboard"


def test_strips_leading_and_trailing_whitespace():
    assert clean_nav("  Batch Scan Centre  ") == "Batch Scan Centre"


def test_multiple_emojis_removed():
    assert clean_nav("🔍📊 Model & Knowledge Base 🛡️") == "Model & Knowledge Base"


def test_empty_string_returns_empty():
    assert clean_nav("") == ""
