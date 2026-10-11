.PHONY: help install install-dev run train preprocess test

help:  ## List available commands
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-12s %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	pip install -r requirements.txt

install-dev:  ## Install runtime + test dependencies
	pip install -r requirements.txt -r requirements-dev.txt

run:  ## Launch the Streamlit dashboard
	streamlit run app.py

preprocess:  ## Rebuild data/processed/sar_dataset.csv
	python preprocessing.py

train:  ## Retrain the XGBoost model (writes model.json + features.pkl)
	python train.py

test:  ## Run the test suite
	python -m pytest -q
