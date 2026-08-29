"""
Regression test: requirements.txt previously had no version bounds on
xgboost or scikit-learn, even though model.pkl already emits a pickle
compatibility UserWarning on load. An unpinned future major release
could silently break the live Streamlit Cloud deployment on next
rebuild. See KNOWN_ISSUES.md for the full writeup.
"""
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def test_xgboost_and_sklearn_are_version_pinned():
    with open(os.path.join(BASE_DIR, "requirements.txt")) as f:
        content = f.read()

    xgboost_line = next((ln for ln in content.splitlines() if ln.strip().startswith("xgboost")), None)
    sklearn_line = next((ln for ln in content.splitlines() if ln.strip().startswith("scikit-learn")), None)

    assert xgboost_line is not None, "xgboost must be listed in requirements.txt"
    assert sklearn_line is not None, "scikit-learn must be listed in requirements.txt"

    assert re.search(r"[<>=]", xgboost_line), (
        f"xgboost must have a version constraint, got: {xgboost_line!r}"
    )
    assert re.search(r"[<>=]", sklearn_line), (
        f"scikit-learn must have a version constraint, got: {sklearn_line!r}"
    )
