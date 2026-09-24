.PHONY: lint test

lint:
	python -m ruff check src tests scripts config

test:
	python -m pytest -q
