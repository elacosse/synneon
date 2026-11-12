# Makefile for easy development workflows.
# CI runs the same checks.

.DEFAULT_GOAL := default

UV_CONFIG_FILE := $(CURDIR)/uv.toml
export UV_CONFIG_FILE

.PHONY: default setup install check lint format typecheck test upgrade build clean

PYTHON_PATHS := src tests
DOC_PATHS := README.md AGENTS.md CHANGELOG.md development.md installation.md publishing.md
YAML_PATHS := .github $(wildcard .copier-answers.yml) .yamllint.yaml docker
CODESPELL_SKIP := --skip="*/node_modules/*,*/dist/*,*/package-lock.json"

default: check

setup: install
	uv run pre-commit install

install:
	uv sync --all-extras --dev

lint:
	uv run codespell $(CODESPELL_SKIP) $(PYTHON_PATHS) $(DOC_PATHS)
	uv run ruff check $(PYTHON_PATHS)
	uv run ruff format --check $(PYTHON_PATHS)
	uv run yamllint --config-file .yamllint.yaml $(YAML_PATHS)

format:
	uv run codespell $(CODESPELL_SKIP) --write-changes $(PYTHON_PATHS) $(DOC_PATHS)
	uv run ruff check --fix $(PYTHON_PATHS)
	uv run ruff format $(PYTHON_PATHS)

# Not part of `check` yet: the existing code has outstanding ty diagnostics.
typecheck:
	uv check --locked

test:
	uv run pytest

check: lint test

upgrade:
	uv lock --upgrade
	uv sync

build:
	uv sync --locked --group build
	uv build --no-build-isolation

clean:
	-rm -rf dist/
	-rm -rf *.egg-info/
	-rm -rf .coverage coverage.xml htmlcov/
	-rm -rf .pytest_cache/
	-rm -rf .ruff_cache/
	-rm -rf .venv/
	-find . -type d -name "__pycache__" -exec rm -rf {} +
