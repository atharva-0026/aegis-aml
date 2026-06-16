# Aegis — AML Compliance Platform

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
