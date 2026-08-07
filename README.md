# Aegis — AML Compliance Platform

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://github.com/atharva-0026/aegis-aml/actions/workflows/tests.yml/badge.svg)

🔗 **Live Demo:** https://aegis-aml.streamlit.app

End-to-end anti-money laundering detection platform built for financial institutions.

## Features
- **XGBoost risk scoring** — trained classifier flags suspicious transactions by probability, with hardcoded rule overrides for high-value transfers
- **RAG-lite narrative generation** — TF-IDF retrieval over a regulatory knowledge base feeds context into Llama 3.1 (via Groq) to draft SAR narratives, with a template-based fallback when the API is unavailable
- **Streamlit terminal-style dashboard** — dark, data-dense UI for single-transaction scans, batch CSV scoring, and model diagnostics

## Stack
`Python` `Streamlit` `XGBoost` `scikit-learn` `Groq (Llama 3.1)` `Plotly`

## Research
Published at **ICAIIHI 2025** — GNN-based AML detection in financial networks. (Note: this repo's production model is XGBoost; the GNN approach was explored in the accompanying research paper, not deployed here.)

## Architecture
Streamlit frontend → XGBoost risk scorer → TF-IDF regulatory retrieval → Llama 3.1 narrative generation (Groq API, with local fallback)

## License
MIT — see [LICENSE](LICENSE)

## Known Issues
See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for a tracked train/inference feature mismatch.

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Testing
```bash
pip install -r requirements-dev.txt
pytest
```
38 tests covering inference, RAG retrieval, model artifacts, and utility functions.
