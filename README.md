# Aegis — AML Compliance Platform

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://github.com/atharva-0026/aegis-aml/actions/workflows/tests.yml/badge.svg)

🔗 **Live Demo:** https://aegis-aml.streamlit.app

End-to-end anti-money laundering detection platform built for financial institutions.

## Features
- **GNN-based transaction risk scoring** — graph neural network models suspicious transaction patterns
- **XGBoost + SHAP** — explainable ML classifier with interpretable risk factors
- **RAG pipeline** — auto-generates SAR narratives using LangChain + ChromaDB + Llama 3.1
- **Bloomberg Terminal-style frontend** — dark, data-dense React UI

## Stack
`Python` `FastAPI` `Streamlit` `LangChain` `ChromaDB` `XGBoost` `React`

## Research
Published at **ICAIIHI 2025** — GNN-based AML detection in financial networks

## Architecture
FastAPI backend → GNN risk scorer → XGBoost classifier → RAG narrative generator → React frontend

## License
MIT — see [LICENSE](LICENSE)

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```
