PY ?= python

.PHONY: all data validate pipeline test clean

all: data validate pipeline test

data:
	$(PY) data/download_data.py

validate:
	$(PY) -m src.validate_lgbm

pipeline:
	$(PY) -m src.pipeline

test:
	$(PY) -m pytest -q

clean:
	rm -rf data/processed reports/figures/*.png reports/tables/*.csv reports/tables/*.md reports/tables/*.json reports/tables/*.txt
