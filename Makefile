.DEFAULT_GOAL := all
sources = fastnltk tests
PYTHON ?= python
NUM_JOBS ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

export CARGO_TERM_COLOR = $(shell (test -t 0 && echo "always") || echo "auto")

# Auto-enable sccache if installed
SCCACHE := $(shell which sccache 2>/dev/null)
ifdef SCCACHE
export RUSTC_WRAPPER := $(SCCACHE)
endif

# ── Installation ──────────────────────────────────────
.PHONY: install
install:  ## Install deps + pre-commit hooks
	pip install -e ".[dev]"
	pre-commit install --install-hooks

.PHONY: install-tools
install-tools:  ## Install perf tools (sccache, nextest, mold)
	cargo install sccache cargo-nextest 2>/dev/null || true
	@# mold on Linux: apt install mold or brew install mold
	@# lld on Windows: choco install llvm

# ── Build ─────────────────────────────────────────────
.PHONY: build-dev
build-dev:  ## Build fast dev version
	maturin develop --uv 2>/dev/null || maturin develop

.PHONY: build-prod
build-prod:  ## Build optimized release
	maturin develop --uv --release 2>/dev/null || maturin develop --release

.PHONY: build-wheel
build-wheel:  ## Build release wheel
	maturin build --release --out dist

.PHONY: build-timing
build-timing:  ## Build with timing report
	CARGO_PROFILE_RELEASE_LTO=fat maturin build --release --out dist 2>&1; \
	  echo "\n📊 Timings saved to target/cargo-timings/"

# ── Formatting ────────────────────────────────────────
.PHONY: format-rs
format-rs:  ## Format Rust only
	cargo fmt || true

.PHONY: format-py
format-py:  ## Format Python only
	ruff check --fix $(sources) || true
	ruff format $(sources) || true

.PHONY: format
format: format-py format-rs  ## Auto-format Rust + Python
	@true

# ── Lint ──────────────────────────────────────────────
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

# ── Test ──────────────────────────────────────────────
.PHONY: test
test:  ## Run Python tests
	$(PYTHON) -m pytest

.PHONY: test-rust
test-rust:  ## Run Rust tests (via nextest if available, fallback to cargo test)
	NEXTEST := $(shell which cargo-nextest 2>/dev/null)
	if [ -n "$$NEXTEST" ]; then \
		cargo nextest run --no-default-features --features parallel; \
	else \
		cargo test --no-default-features --features parallel; \
	fi

.PHONY: test-all
test-all: test test-rust  ## Run all tests (sequential)

.PHONY: test-par
test-par:  ## Run all tests in parallel
	@echo "Running Python + Rust tests in parallel..."
	@$(MAKE) test > /tmp/test-py.log 2>&1 & \
	$(MAKE) test-rust > /tmp/test-rs.log 2>&1 & \
	wait; \
	cat /tmp/test-py.log; \
	cat /tmp/test-rs.log

# ── Benchmarks ────────────────────────────────────────
.PHONY: bench
bench: build-prod  ## Run benchmarks
	$(PYTHON) -m pytest benchmarks/ --benchmark-only --benchmark-sort=name 2>/dev/null \
	  || echo "No benchmarks/ directory or no benchmark tests found"

.PHONY: bench-compare
bench-compare: build-prod  ## Run benchmarks with comparison
	$(PYTHON) -m pytest benchmarks/ --benchmark-compare --benchmark-sort=name 2>/dev/null \
	  || echo "No benchmarks/ directory found"

# ── Clean ─────────────────────────────────────────────
.PHONY: clean
clean:  ## Remove build artifacts
	rm -rf `find . -name __pycache__`
	rm -f `find . -type f -name '*.py[co]'`
	rm -rf .pytest_cache build dist
	rm -f fastnltk/_rust/*.so fastnltk/_rust/*.dll fastnltk/_rust/*.dylib 2>/dev/null; true

.PHONY: clean-cargo
clean-cargo:  ## Remove cargo build artifacts (keeps deps)
	cargo clean -p fastnltk 2>/dev/null || true

.PHONY: clean-all
clean-all: clean clean-cargo  ## Full clean
	cargo clean

# ── CI standard ───────────────────────────────────────
.PHONY: all
all: build-dev lint test  ## CI standard check set

# ── Utility ───────────────────────────────────────────
.PHONY: check-rs
check-rs:  ## Fast Rust compilation check
	cargo check --no-default-features --features parallel

.PHONY: audit
audit:  ## Dependency audit
	cargo deny check 2>/dev/null || cargo audit 2>/dev/null || echo "Install cargo-deny: cargo install cargo-deny"

.PHONY: outdated
outdated:  ## Check outdated deps
	cargo outdated 2>/dev/null || echo "Install: cargo install cargo-outdated"

.PHONY: help
help:  ## Display this message
	@grep -E '^.PHONY: .* ## .*$$' $(MAKEFILE_LIST) | \
		sort | \
		awk 'BEGIN {FS = ".PHONY: |## "}; {printf "\033[36m%-19s\033[0m %s\n", $$2, $$3}'
