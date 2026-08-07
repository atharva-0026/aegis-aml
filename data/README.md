# Data

- `creditcard.csv` (repo root, gitignored) — raw transaction data expected by `preprocessing.py`. Not committed; place it in the project root before running preprocessing.
- `processed/` — output of `preprocessing.py`, used directly by `train.py`

Run `python preprocessing.py` to regenerate processed data from raw.

Note: `preprocessing.py` also creates an empty `data/raw/` directory on
startup, but currently reads `creditcard.csv` from the repo root, not
from `data/raw/`. That directory is unused — see KNOWN_ISSUES.md.
