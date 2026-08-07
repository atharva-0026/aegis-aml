"""
Regression test: README previously claimed FastAPI, React, GNN, ChromaDB,
and LangChain — none of which exist in this codebase. Guards against the
README drifting from the actual stack again.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _read_readme():
    with open(os.path.join(BASE_DIR, "README.md"), encoding="utf-8") as f:
        return f.read()


def test_readme_does_not_claim_unused_frameworks():
    content = _read_readme()
    unused_claims = ["FastAPI", "ChromaDB", "LangChain", "React"]
    found = [term for term in unused_claims if term in content]
    assert not found, f"README claims unused technologies: {found}"


def test_readme_mentions_actual_stack():
    content = _read_readme()
    for term in ["Streamlit", "XGBoost", "Groq"]:
        assert term in content, f"README should mention {term}, the actual stack"
