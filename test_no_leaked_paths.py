"""
Repo-wide regression test: guards against leaked local developer paths
(e.g. C:\\Users\\someone\\Desktop\\...) ever being committed again.
See KNOWN_ISSUES.md / fix history for the original incident.
"""
import os
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LEAK_PATTERNS = ["C:\\Users", "/Users/", "Desktop\\EDI", "HP\\Desktop"]


def _tracked_python_files():
    return [
        f for f in glob.glob(os.path.join(BASE_DIR, "*.py"))
        if os.path.isfile(f) and os.path.basename(f) != "test_no_leaked_paths.py"
    ]


def test_no_leaked_local_paths_in_source():
    offenders = []
    for path in _tracked_python_files():
        with open(path, encoding="utf-8", errors="ignore") as fh:
            content = fh.read()
        for pattern in LEAK_PATTERNS:
            if pattern in content:
                offenders.append((os.path.basename(path), pattern))

    assert not offenders, f"Leaked local paths found: {offenders}"
