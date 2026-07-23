# Changelog

## Unreleased
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
