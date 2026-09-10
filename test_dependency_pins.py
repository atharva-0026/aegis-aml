"""
Regression test: requirements.txt previously had no version bounds on
xgboost or scikit-learn. That mattered a lot more before the model
storage format migration (model.pkl -> model.json, see
KNOWN_ISSUES.md) - the pins are now defense-in-depth rather than the
only thing preventing a live-deployment break, but still worth
keeping in place.
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
