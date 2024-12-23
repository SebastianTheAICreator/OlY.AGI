# Makefile for OLy AGI Framework

.PHONY: install test format lint clean docs

install:
	poetry install

test:
	pytest tests/ --cov=oly --cov-report=term-missing

format:
	black .
	isort .

lint:
	mypy .
	black . --check
	isort . --check

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

docs:
	mkdocs build

serve-docs:
	mkdocs serve

dist: clean
	poetry build
