# Equity Analytics — common tasks
.PHONY: help setup run dashboard notebook refresh clean

PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin

help:
	@echo "  make setup      create the virtualenv and install dependencies"
	@echo "  make run        run the full pipeline (charts + reports)"
	@echo "  make dashboard  launch the dashboard at http://127.0.0.1:8050"
	@echo "  make notebook   open the EDA notebook"
	@echo "  make refresh    re-download prices, ignoring the cache"
	@echo "  make clean      remove generated outputs and caches"

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --quiet --upgrade pip
	$(BIN)/pip install -r requirements.txt
	@echo "Done. Next: make run"

run:
	$(BIN)/python main.py

refresh:
	$(BIN)/python main.py --refresh

dashboard:
	$(BIN)/python dashboard.py

notebook:
	$(BIN)/jupyter lab notebooks/EDA.ipynb

clean:
	rm -rf outputs/charts/*.png reports/*.csv reports/*.txt reports/*.xlsx
	rm -rf data/cache data/enriched
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
