.DEFAULT_GOAL := all
sources = fastnltk tests
PYTHON ?= python

export CARGO_TERM_COLOR = $(shell (test -t 0 && echo "always") || echo "auto")

.PHONY: install
install:  ## Install deps + pre-commit hooks
	pip install -e ".[dev]"
	pre-commit install --install-hooks

.PHONY: build-dev
build-dev:  ## Build fast dev version
	maturin develop --uv 2>/dev/null || maturin develop

.PHONY: build-prod
build-prod:  ## Build optimized release
	maturin develop --uv --release 2>/dev/null || maturin develop --release

.PHONY: build-wheel
build-wheel:  ## Build release wheel
	maturin build --release --out dist

.PHONY: format
format:  ## Auto-format Rust + Python
	ruff check --fix $(sources) || true
	ruff format $(sources) || true
	cargo fmt || true

.PHONY: lint-python
lint-python:  ## Lint Python source
	ruff check $(sources)
	ruff format --check $(sources)

.PHONY: lint-rust
lint-rust:  ## Lint Rust source
	cargo fmt --all -- --check
	cargo clippy --tests -- -D warnings 2>/dev/null || cargo clippy --tests

.PHONY: lint
lint: lint-python lint-rust  ## Lint all

.PHONY: test
test:  ## Run Python tests
	$(PYTHON) -m pytest

.PHONY: test-rust
test-rust:  ## Run Rust tests
	cargo test

.PHONY: test-all
test-all: test test-rust  ## Run all tests

.PHONY: clean
clean:  ## Remove build artifacts
	rm -rf `find . -name __pycache__`
	rm -f `find . -type f -name '*.py[co]'`
	rm -rf .pytest_cache build dist
	rm -f fastnltk/_rust/*.so fastnltk/_rust/*.dll fastnltk/_rust/*.dylib 2>/dev/null; true

.PHONY: all
all: build-dev lint test  ## CI standard check set

.PHONY: help
help:  ## Display this message
	@grep -E '^.PHONY: .* ## .*$$' $(MAKEFILE_LIST) | \
		sort | \
		awk 'BEGIN {FS = ".PHONY: |## "}; {printf "\033[36m%-19s\033[0m %s\n", $$2, $$3}'
