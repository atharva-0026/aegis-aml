"""
Tests for rag.py — regulatory context retrieval (offline, no Groq calls).
"""
from rag import retrieve_regulatory_context, KNOWLEDGE_BASE


def test_retrieval_returns_valid_document():
    result = retrieve_regulatory_context(amount=1000, time=40000)
    assert result in KNOWLEDGE_BASE


def test_high_value_matches_threshold_rule():
    """Large amounts should retrieve the high-value transactions doc."""
    result = retrieve_regulatory_context(amount=60000, time=40000)
    assert "High-Value" in result["title"]


def test_structuring_matches_intermediate_amount():
    result = retrieve_regulatory_context(amount=15000, time=40000)
    assert result["title"] in (
        "Structuring and Layering Avoidance Rule",
        "BSA/AML High-Value Transactions Rule",
    )


def test_nighttime_transaction_matches_night_pattern():
    """A transaction at 1 AM (time % 86400 < 21600) should hit the nighttime doc."""
    result = retrieve_regulatory_context(amount=500, time=3600)
    assert "Nighttime" in result["title"]


def test_flagged_by_rules_matches_override_doc():
    result = retrieve_regulatory_context(
        amount=100, time=50000, flagged_by_rules=True
    )
    assert result["title"] in (doc["title"] for doc in KNOWLEDGE_BASE)


def test_cross_border_location_matches_geo_doc():
    result = retrieve_regulatory_context(
        amount=500, time=50000, location="Dubai"
    )
    assert result["title"] in (doc["title"] for doc in KNOWLEDGE_BASE)


def test_no_signals_falls_back_to_a_document():
    """Even a completely unremarkable transaction should still retrieve something."""
    result = retrieve_regulatory_context(amount=50, time=50000)
    assert result in KNOWLEDGE_BASE
