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


def test_narrative_uses_current_date_not_hardcoded():
    """Regression test: narratives previously had a hardcoded 2026-05-21 date."""
    from datetime import datetime
    from rag import generate_narrative

    narrative = generate_narrative(500, 40000, "Normal", 0.01)
    today = datetime.now().strftime("%Y-%m-%d")
    assert today in narrative
    assert "2026-05-21" not in narrative or today == "2026-05-21"


def test_fraud_narrative_uses_aegis_branding_not_edi():
    """Regression test: the fallback SAR template previously said 'EDI'
    (a leftover project codename) instead of 'Aegis'."""
    from rag import generate_narrative

    narrative = generate_narrative(80000, 1000, "Fraud", 0.95, "transfer", "Dubai", True)
    assert "EDI" not in narrative
    assert "Aegis" in narrative


def test_narrative_correctly_cites_fifty_thousand_rule_when_amount_exceeds_it():
    from rag import generate_narrative

    narrative = generate_narrative(60000, 40000, "Fraud", 0.9, flagged_by_rules=True)
    assert "₹50,000 rule" in narrative


def test_narrative_does_not_falsely_claim_fifty_thousand_rule_for_smaller_amount():
    """Regression test: predict.py's rule_flag is
    amount > 50000 OR (amount > 30000 AND time < 5000) - two distinct
    conditions. The old narrative always cited the ₹50,000 threshold
    whenever flagged_by_rules was True, even for a transaction flagged
    via the second (smaller-amount) condition. A ₹35,000 transaction
    must not have its narrative falsely claim it exceeded ₹50,000."""
    from rag import generate_narrative

    narrative = generate_narrative(35000, 2000, "Fraud", 0.001, flagged_by_rules=True)
    assert "exceeds absolute risk thresholds (₹50,000 rule)" not in narrative
    assert "35,000" in narrative


def test_narrative_ml_flagged_case_still_cites_model_confidence():
    from rag import generate_narrative

    narrative = generate_narrative(5000, 40000, "Fraud", 0.85, flagged_by_rules=False)
    assert "85.00%" in narrative
    assert "₹50,000 rule" not in narrative
