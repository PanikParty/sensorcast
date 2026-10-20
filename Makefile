VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: help setup bench quick test smoke report clean

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

setup: ## create venv and install package + model dependencies
	uv venv --python 3.12 $(VENV)
	uv pip install -e ".[models,dev]"

bench: ## full benchmark -> results/ + REPORT.md (~8 min, CPU only)
	$(PY) scripts/run_benchmark.py --out results

quick: ## fast slice: 24h horizon, two models
	$(PY) scripts/run_benchmark.py --out results/quick --horizons 24 \
		--models timestral:209M baseline:seasonal_naive

smoke: ## verify every model loads and predicts (no scoring)
	$(PY) scripts/smoke_models.py

test: ## unit tests
	$(PY) -m pytest -q

clean: ## remove caches and quick results
	rm -rf results/quick .pytest_cache **/__pycache__
