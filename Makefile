.PHONY: setup test lint simulate real extract report

setup:
	pip install -e ".[dev,bigquery,report]"

test:
	pytest

lint:
	ruff check src tests

simulate:
	journey-attribution --mode simulate

real:
	journey-attribution --mode real

extract:
	python -m journey_attribution.ingestion.bigquery --project $(PROJECT)

report:
	journey-attribution --mode real --report reports/report.html
