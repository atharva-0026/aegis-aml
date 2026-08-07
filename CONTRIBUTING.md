# Contributing

## Local development
```bash
git clone https://github.com/atharva-0026/aegis-aml.git
cd aegis-aml
pip install -r requirements.txt
streamlit run app.py
```

## Project layout
- `app.py` — Streamlit frontend
- `train.py` / `train_utils.py` — model training pipeline
- `predict.py` — inference on new transactions
- `preprocessing.py` — feature engineering
- `rag.py` — SAR narrative generation
- `nav_utils.py` — small UI helper functions
- `test_*.py` — pytest suite (run with `pytest`)

## Guidelines
- Keep functions documented with docstrings
- Run the app locally before opening a PR
- Never commit local absolute paths (e.g. `C:\Users\...`, `/Users/...`) as
  fallback values in source — use relative paths or environment variables
  instead. `test_no_leaked_paths.py` guards against this in CI.
