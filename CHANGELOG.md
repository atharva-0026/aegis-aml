# Changelog

## Unreleased
- Added regression tests for data/raw fix and README accuracy guard
- Fixed data/raw/ directory being created but never actually read by preprocessing.py
- Rewrote README to match actual stack (Streamlit not React, XGBoost not GNN, no FastAPI/ChromaDB/LangChain) — see KNOWN_ISSUES.md
- Fixed leftover "EDI" branding in SAR fallback template (now correctly says Aegis)
- Fixed silently swallowed Groq API errors — now logged to stderr on fallback
- Fixed unseeded random.choice() calls in preprocessing.py breaking full dataset reproducibility
- Removed leaked personal Windows dev paths (C:\Users\HP\Desktop\...) from preprocessing.py and train.py
- Extracted resolve_dataset_path and FEATURE_COLUMNS into train_utils.py for testability
- Added regression test guarding against future local-path leaks in source
- Resolved train/inference feature mismatch: amount_bin and amount_ratio now use exact training-derived constants (see KNOWN_ISSUES.md)
- Flagged real train/inference feature mismatch in amount_bin and amount_ratio (see KNOWN_ISSUES.md)
- Added type hints to predict_transaction and clean_nav
- Added pytest.ini config, CODEOWNERS
- Extracted clean_nav into nav_utils.py with dedicated unit tests
- Fixed hardcoded date (2026-05-21) in SAR narrative templates
- Added ValueError validation for negative amount/time in predict_transaction
- Added 18-test suite covering predict.py, rag.py, and model artifacts
- Added GitHub Actions CI workflow to run tests on push/PR
- Added MIT LICENSE and .env.example
- Added shield favicon to Streamlit page
- Added README badges and license link
- Added PR and issue templates
- Expanded `.gitignore` to cover logs and cache directories
- Added module docstrings to app.py, train.py, rag.py, preprocessing.py, predict.py
- Added CONTRIBUTING.md with local dev setup and project layout
- Minor housekeeping and documentation touch-ups
