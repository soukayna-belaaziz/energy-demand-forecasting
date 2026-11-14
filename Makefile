PY ?= python

.PHONY: all full data validate pipeline notebook test clean

# Default: download data, run the pipeline (incl. SARIMAX spec validation), run tests.
all: data pipeline test

# Also re-run the LightGBM feature-set validation and re-execute the walkthrough notebook.
full: data validate pipeline notebook test

data:
	$(PY) data/download_data.py

validate:
	$(PY) -m src.validate_lgbm

pipeline:
	$(PY) -m src.pipeline

notebook:
	jupyter nbconvert --to notebook --execute --inplace notebooks/walkthrough.ipynb

test:
	$(PY) -m pytest -q

clean:
	rm -rf data/processed reports/figures/*.png reports/tables/*.csv reports/tables/*.md reports/tables/*.json reports/tables/*.txt
