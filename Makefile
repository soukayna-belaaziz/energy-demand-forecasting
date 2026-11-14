PY ?= python

.PHONY: all data pipeline test clean

all: data pipeline test

data:
	$(PY) data/download_data.py

pipeline:
	$(PY) -m src.pipeline

test:
	$(PY) -m pytest -q

clean:
	rm -rf data/processed reports/figures/*.png reports/tables/*
