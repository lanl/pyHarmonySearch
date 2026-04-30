# Makefile for pyHarmonySearch
# Provides convenient commands for testing, linting, and running examples

.PHONY: help install install-dev test test-fast test-unit test-integration test-functional test-property test-cov test-all lint format type-check clean coverage-html coverage-report examples

# Default target - show help
help:
	@echo "pyHarmonySearch - Available Make Targets"
	@echo "=========================================="
	@echo ""
	@echo "Installation:"
	@echo "  make install          - Install package in editable mode (pip install -e .)"
	@echo "  make install-dev      - Install with dev dependencies (pip install -e '.[dev]')"
	@echo ""
	@echo "Testing:"
	@echo "  make test             - Run all tests with coverage"
	@echo "  make test-fast        - Run unit tests only with fast failure (-x)"
	@echo "  make test-unit        - Run unit tests"
	@echo "  make test-integration - Run integration tests"
	@echo "  make test-functional  - Run functional tests"
	@echo "  make test-property    - Run property-based tests"
	@echo "  make test-cov         - Run tests with detailed coverage report"
	@echo "  make test-all         - Run tests in parallel (-n auto)"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             - Run ruff linter"
	@echo "  make format           - Format code with ruff"
	@echo "  make type-check       - Run mypy type checker"
	@echo ""
	@echo "Coverage:"
	@echo "  make coverage-html    - Generate HTML coverage report"
	@echo "  make coverage-report  - Print coverage report to terminal"
	@echo ""
	@echo "Examples:"
	@echo "  make examples         - Run all example scripts"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            - Remove generated files (.pytest_cache, .coverage, etc.)"
	@echo ""

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Testing targets
test:
	@if [ -d tests ]; then \
		pytest --cov=pyharmonysearch --cov-report=term-missing; \
	else \
		echo "No tests directory found. Run 'make examples' to verify functionality."; \
		exit 1; \
	fi

test-fast:
	@if [ -d tests ]; then \
		pytest tests/unit -x; \
	else \
		echo "No tests directory found. Run 'make examples' to verify functionality."; \
		exit 1; \
	fi

test-unit:
	@if [ -d tests/unit ]; then \
		pytest tests/unit -v; \
	else \
		echo "No tests/unit directory found."; \
		exit 1; \
	fi

test-integration:
	@if [ -d tests/integration ]; then \
		pytest tests/integration -v; \
	else \
		echo "No tests/integration directory found."; \
		exit 1; \
	fi

test-functional:
	@if [ -d tests/functional ]; then \
		pytest tests/functional -v; \
	else \
		echo "No tests/functional directory found."; \
		exit 1; \
	fi

test-property:
	@if [ -d tests/property ]; then \
		pytest tests/property -v; \
	else \
		echo "No tests/property directory found."; \
		exit 1; \
	fi

test-cov:
	@if [ -d tests ]; then \
		pytest --cov=pyharmonysearch --cov-report=term-missing --cov-report=html; \
	else \
		echo "No tests directory found. Run 'make examples' to verify functionality."; \
		exit 1; \
	fi

test-all:
	@if [ -d tests ]; then \
		pytest -n auto --cov=pyharmonysearch; \
	else \
		echo "No tests directory found. Run 'make examples' to verify functionality."; \
		exit 1; \
	fi

# Code quality targets
lint:
	ruff check pyharmonysearch/ examples/

format:
	ruff format pyharmonysearch/ examples/

type-check:
	mypy pyharmonysearch/

# Coverage targets
coverage-html:
	@if [ -d tests ]; then \
		pytest --cov=pyharmonysearch --cov-report=html; \
		echo "Coverage report generated in htmlcov/index.html"; \
	else \
		echo "No tests directory found."; \
		exit 1; \
	fi

coverage-report:
	@if [ -f .coverage ]; then \
		coverage report -m; \
	else \
		echo "No .coverage file found. Run 'make test' or 'make test-cov' first."; \
		exit 1; \
	fi

# Examples target - runs all example scripts
examples:
	@echo "Running pyHarmonySearch examples..."
	@echo "===================================="
	@echo ""
	@for script in examples/*.py; do \
		echo "Running $$script..."; \
		python "$$script" || exit 1; \
		echo ""; \
	done
	@echo "All examples completed successfully!"

# Cleanup target
clean:
	@echo "Cleaning up generated files..."
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf *.egg-info
	rm -rf dist
	rm -rf build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.so" -delete
	@echo "Cleanup complete!"
