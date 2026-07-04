# Billion-Scale Vector Search — developer entrypoints.
# Offline, CPU-only, deterministic (seed=42). No pip installs required if the
# shared portfolio stack (numpy/sklearn/pandas/matplotlib/scipy/pytest) is present.

PY ?= python
export MPLBACKEND = Agg
export PYTHONPATH = src

ROWS ?= 100000
DIM  ?= 128

.PHONY: help setup data build test bench screenshots all clean

help:
	@echo "make setup       - install requirements (only if stack missing)"
	@echo "make data        - stream a demo dataset to data/base (ROWS=$(ROWS))"
	@echo "make build       - build one IVF-PQ index from a stream and report stats"
	@echo "make test        - run the pytest suite (real assertions)"
	@echo "make bench       - full benchmark sweep -> benchmarks/results.csv + RESULTS.md"
	@echo "make screenshots - render assets/*.png from benchmark results"
	@echo "make all         - test + bench + screenshots"
	@echo "make clean       - remove generated data and caches"

setup:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) scripts/generate_data.py --rows $(ROWS) --dim $(DIM) --out data/base

build:
	$(PY) scripts/build_index.py --rows $(ROWS) --dim $(DIM) --nlist 1024 --m 16

test:
	$(PY) -m pytest tests/ -q

bench:
	$(PY) scripts/bench.py

screenshots:
	$(PY) scripts/make_screenshots.py

all: test bench screenshots

clean:
	rm -rf data/* ; touch data/.gitkeep
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
