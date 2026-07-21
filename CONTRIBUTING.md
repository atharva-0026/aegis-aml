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
- `train.py` — model training pipeline
- `predict.py` — inference on new transactions
- `preprocessing.py` — feature engineering
- `rag.py` — SAR narrative generation

## Guidelines
- Keep functions documented with docstrings
- Run the app locally before opening a PR
