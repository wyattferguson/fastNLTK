# Contributing to fastNLTK

Thank you for your interest in fastNLTK! This document covers everything
you need to know to contribute effectively.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Benchmarking](#benchmarking)
- [Pull Request Process](#pull-request-process)
- [Adding a New Module](#adding-a-new-module)
- [Python Shim Guidelines](#python-shim-guidelines)
- [Rust Code Guidelines](#rust-code-guidelines)
- [CI Pipeline](#ci-pipeline)
- [Release Process](#release-process)
- [Getting Help](#getting-help)

## Code of Conduct

This project follows the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct).
All contributors are expected to uphold it. Be respectful, inclusive, and constructive.

## Getting Started

### Prerequisites

- **Python 3.8+** — [python.org](https://python.org)
- **Rust 1.80+** — [rustup.rs](https://rustup.rs)
- **NLTK data** — `python -m nltk.downloader punkt averaged_perceptron_tagger wordnet`

### First-time Setup

```bash
git clone https://github.com/your/fastnltk
cd fastnltk

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dev dependencies
pip install -e ".[dev,test,lint]"

# Install maturin for Rust compilation
pip install maturin

# Build + install the Rust extension in development mode
maturin develop --release

# Verify everything works
python -c "from fastnltk import word_tokenize; print(word_tokenize('Hello, world!'))"
# Expected: ['Hello', ',', 'world', '!']
```

## Development Setup

### Building

```bash
# Debug build (fast compile, for development)
maturin develop

# Release build (optimized, for benchmarks)
maturin develop --release

# Build wheel
maturin build --release
```

The Rust extension is compiled as `fastnltk._rust`. You can rebuild
it independently with cargo:

```bash
cargo build                     # Debug
cargo build --release           # Release
```

### Running Tests

```bash
# Rust tests (fast — ~2s)
cargo test --all-targets

# Run a specific Rust test
cargo test test_tokenize_treebank

# Python tests (require maturin build first)
pytest tests/ -v

# Run a specific Python test
pytest tests/test_tokenize.py -v -k "test_treebank"

# Both combined
cargo test --all-targets && pytest tests/ -q
```

### Linting

```bash
# Rust formatting
cargo fmt --all -- --check

# Rust linting (same flags as CI)
cargo clippy --all-targets -- \
    -D clippy::correctness \
    -D clippy::suspicious \
    -D clippy::perf \
    -W clippy::style \
    -W clippy::complexity

# Python linting
ruff check fastnltk/ tests/

# Rust doc links
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items
```

## Project Structure

```
fastnltk/
├── src/                          # Rust source (PyO3 extension)
│   ├── lib.rs                    # Module registration, global allocator
│   ├── error.rs                  # FastNltkError enum (thiserror)
│   ├── tree.rs                   # Tree data structure
│   ├── sem/                      # Semantics (split from sem.rs)
│   │   ├── expression.rs         #   Expression/Type enums
│   │   ├── parse.rs              #   Recursive descent parser
│   │   ├── evaluate.rs           #   Model evaluation
│   │   └── mod.rs                #   Re-exports + register
│   ├── tokenize/                 # All tokenizers
│   ├── tag/                      # POS taggers
│   │   ├── sequential/           #   Backoff taggers (split from sequential.rs)
│   │   │   ├── mod.rs            #     DefaultTagger
│   │   │   └── taggers.rs        #     Unigram/Bigram/Trigram/Affix/Regexp
│   │   ├── perceptron.rs         #   Averaged perceptron
│   │   ├── tnt.rs                #   Trigram HMM
│   │   ├── hmm.rs                #   Hidden Markov Model
│   │   └── mod.rs                #   Module registration
│   ├── lm/                       # Language models (split from lm.rs)
│   │   ├── mod.rs                #   MLE/Lidstone/Laplace
│   │   ├── kneser_ney.rs         #   Kneser-Ney smoothing
│   │   ├── witten_bell.rs        #   Witten-Bell smoothing
│   │   └── stupid_backoff.rs     #   Stupid backoff
│   ├── probability/              # Probability distributions (split from probability.rs)
│   │   ├── mod.rs                #   FreqDist + ConditionalFreqDist
│   │   └── dist.rs               #   MLEProbDist + LaplaceProbDist
│   ├── ccg/                      # Combinatory Categorial Grammar
│   ├── classify/                 # NaiveBayes, Maxent, TextCat
│   ├── inference/                # Tableau + Resolution provers
│   ├── stem/                     # All stemmers
│   ├── metrics/                  # Association, segmentation, distance
│   ├── drt.rs                    # Discourse Representation Theory
│   ├── parse.rs                  # CFG + Earley parsing
│   ├── chunk.rs                  # Regexp chunking
│   ├── sentiment.rs              # VADER sentiment
│   ├── collocations.rs           # Ngram collocation finders
│   ├── cluster.rs                # K-means clustering
│   ├── chat.rs                   # Eliza chatbot
│   ├── translate.rs              # BLEU score
│   ├── corpus.rs                 # Corpus reader wrappers
│   ├── data.rs                   # Resource finder
│   ├── prelude.rs                # Common imports
│   └── util/                     # Regex cache, string utilities
├── fastnltk/                     # Python shim layer
│   ├── __init__.py               # Re-exports from _rust + NLTK fallback
│   ├── tokenize.py               # Tokenizer wrappers
│   ├── tag.py                    # Tagger wrappers
│   ├── stem.py                   # Stemmer wrappers
│   ├── ...                       # (21 shim files total)
├── tests/                        # Python integration tests
│   ├── test_core.py              # Core functionality
│   ├── test_tokenize.py          # Tokenization tests
│   ├── test_tag.py               # Tagging tests
│   ├── test_tree.py              # Tree tests
│   ├── ...                       # (18 test files total)
├── benchmarks/                   # Benchmark harness
│   ├── harness.py                # Base harness (regression detection)
│   ├── bench_suite.py            # All benchmark definitions
│   └── run.py                    # CLI entry point
├── fuzz/                         # Fuzz targets (cargo-fuzz)
│   ├── fuzz_targets/
│   │   ├── ccg_parse.rs          #   CCG parser fuzz
│   │   └── drs_parse.rs          #   DRS parser fuzz
│   └── Cargo.toml
├── Cargo.toml                    # Rust manifest
├── pyproject.toml                # Python build config + tool settings
├── deny.toml                     # cargo-deny policy
├── rustfmt.toml                  # Rust formatter config
└── .github/workflows/
    └── quality.yml               # CI: fmt, clippy, doc, test, audit, deny, hack, bench
```

## Coding Standards

### Rust

- **Formatting**: `cargo fmt` (run before every commit). Config in `rustfmt.toml`.
- **Style**: Follow clippy at the `correctness` + `suspicious` + `perf` + `style` + `complexity` level.
  Pedantic and nursery warnings are informational.
- **Unsafe code**: `unsafe_code` is denied at the crate level. Zero unsafe code allowed.
- **Unwraps**: No `unwrap()` in production paths. Use `Result<T, FastNltkError>` or
  `PyResult<T>` with `?` propagation. Use `.expect("...")` only when the failure
  indicates a programming error (invariant violation).
- **Error types**: Use `FastNltkError` (thiserror enum) for Rust-internal errors.
  Use `PyErr` / `PyValueError` for Python-facing errors.
- **Documentation**: All `pub` items must have doc comments. Module-level `//!` docs
  should describe the module purpose and NLTK equivalent.
- **Doc links**: Avoid bare `[bracketed]` text that rustdoc may interpret as links.
  Use backticks `` `like_this` `` for code references, or escape with `\[` `\]`.
- **Imports**: Group by: std → external crates → crate-local. Use `use` not `pub use`
  for internal items.
- **Naming**: Follow Rust conventions. Python-facing items use NLTK naming
  (e.g., `N()` not `n()` for total count) with `#[pyo3(name = "N")]` or `#[allow(non_snake_case)]`.

### Python

- **Formatting**: `ruff format` (run before every commit). Line length 100.
- **Linting**: `ruff check`. No lint errors.
- **Type hints**: All function signatures must have type annotations.
- **Shim pattern**: Each function tries `from fastnltk._rust import ...` first,
  falls back to `from nltk import ...` with a try/except ImportError block.
- **No logic in shims**: Shim files should be thin re-export wrappers.
  All algorithmic code goes in Rust.

### Pre-commit Hooks (Recommended)

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
set -e
cargo fmt --all -- --check || (echo "❌ cargo fmt failed"; exit 1)
cargo clippy --all-targets -- -D clippy::correctness -D clippy::suspicious -D clippy::perf || (echo "❌ clippy failed"; exit 1)
ruff check fastnltk/ tests/ || (echo "❌ ruff failed"; exit 1)
```

## Testing

### Test Organization

- **Rust tests**: Inline `#[cfg(test)] mod tests` in each source file.
  These are fast unit tests (~0.2s total).
- **Python tests**: In `tests/` directory, organized by module.
  These are integration tests that exercise the compiled extension.

### Writing Rust Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_something() {
        let result = my_function("input");
        assert_eq!(result, expected);
    }
}
```

Guidelines:
- Every non-trivial function should have at least one test.
- Test edge cases: empty input, single element, large input, invalid input.
- For algorithms, test against known NLTK output to verify correctness.
- Use `assert_eq!` and `assert!` with descriptive messages for failures.

### Writing Python Tests

```python
def test_something():
    """Descriptive docstring explaining what's being tested."""
    from fastnltk import my_function
    result = my_function("input")
    assert result == expected
```

Guidelines:
- Tests go in `tests/test_<module>.py`.
- Name tests with `test_<function>_<scenario>` pattern.
- Use `pytest` fixtures for test data where appropriate.
- Mark tests that need NLTK data with `@pytest.mark.skipif(..., reason="needs NLTK data")`.

### Running Everything

```bash
# Full test suite
cargo test --all-targets && pytest tests/ -v

# With coverage (requires cargo-llvm-cov + coveragepy)
cargo llvm-cov --all-targets
coverage run -m pytest tests/
coverage report
```

## Benchmarking

### Running Benchmarks

```bash
# Build release first (debug builds are not representative)
maturin develop --release

# Run all benchmarks
python -m benchmarks.run

# Run and save results
python -m benchmarks.run --save

# Compare against a baseline (detect regressions)
python -m benchmarks.run --regression results/latest.json

# CI mode: run + save + check regressions
python -m benchmarks.run --ci --threshold 0.25
```

### Adding a New Benchmark

1. Add a function to `benchmarks/bench_suite.py` that returns a `BenchResult`.
2. The harness automatically discovers functions prefixed with `bench_`.
3. The benchmark is added to the 12-benchmark CI suite automatically.

```python
def bench_my_function() -> BenchResult:
    """Benchmark my_function on test data."""
    from fastnltk import my_function
    import nltk   # for comparison (optional)

    data = fixture("test_data.txt")  # from benchmarks/data/
    nltk_time = _median_time(lambda: nltk.my_function(data), iterations=30)
    fast_time = _median_time(lambda: my_function(data), iterations=30)
    speedup = nltk_time / fast_time if fast_time > 0 else 0.0
    return BenchResult(
        name="MyFunction.method",
        group="my_module",
        nltk_ms=nltk_time * 1000,
        fast_ms=fast_time * 1000,
        speedup=speedup,
    )
```

### Regression Detection

The CI benchmark job uses a 25% threshold. If any benchmark's fastNLTK time
increases by more than 25% compared to the saved baseline, the CI job fails.
This prevents performance regressions from being merged.

## Pull Request Process

1. **Create an issue** first describing the change (unless it's a trivial fix).
2. **Fork the repository** and create a feature branch.
3. **Make your changes**, following the coding standards above.
4. **Run the full test suite** and ensure all tests pass.
5. **Run benchmarks** if your change could affect performance.
6. **Update documentation** if your change adds or modifies public API.
7. **Create a pull request** with a clear description of what changed and why.
8. **Address CI feedback** — all 8 CI jobs must pass before merge.

### PR Checklist

```
□ cargo fmt --all -- --check passes
□ cargo clippy --all-targets -- -D correctness -D suspicious -D perf passes (0 errors)
□ RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items passes
□ cargo test --all-targets passes
□ pytest tests/ -q passes
□ python -m benchmarks.run --save passes (no regressions)
□ ruff check fastnltk/ tests/ passes
□ New code has tests
□ New public API has doc comments
□ New benchmarks added if relevant
□ BENCHMARKS.md updated if performance changed
```

## Adding a New Module

### Adding a Rust module

1. Create `src/<module>/` (or `src/<module>.rs` for simple modules).
2. Register types with PyO3 in the module's `pub fn register_module()`.
3. Add `pub mod <module>;` to `src/lib.rs` and call its `register_module()`.
4. Add Python shim file `fastnltk/<module>.py` following the shim pattern below.
5. Add tests in `tests/test_<module>.py`.
6. Run the full test suite.

### PyO3 Class Registration Pattern

```rust
#[pyclass(name = "MyClass", module = "fastnltk._rust")]
pub struct MyClass {
    inner: SomeType,
}

#[pymethods]
impl MyClass {
    #[new]
    fn new() -> Self { MyClass { inner: SomeType::new() } }

    fn do_stuff(&self, input: String) -> PyResult<String> {
        self.inner.do_stuff(&input).map_err(PyValueError::new_err)
    }
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MyClass>()?;
    Ok(())
}
```

## Python Shim Guidelines

Each Python shim follows this pattern:

```python
"""fastNLTK <module> — Rust-accelerated.

Re-exports Rust implementations with NLTK fallback.
"""

try:
    from fastnltk._rust import MyClass, my_function
except ImportError:
    import warnings
    warnings.warn("fastnltk._rust not found, using NLTK fallback")
    from nltk.my_module import MyClass, my_function  # type: ignore[assignment]
```

The `__init__.py` aggregates all shims:

```python
from fastnltk.tokenize import word_tokenize, sent_tokenize
from fastnltk.tag import pos_tag
# ... etc
```

## Rust Code Guidelines

### Error Handling

```rust
// GOOD: propagate with ?
fn do_thing(input: &str) -> PyResult<String> {
    let parsed = parse(input).map_err(PyValueError::new_err)?;
    Ok(parsed.to_string())
}

// GOOD: expect for invariant violations (with context)
let idx = labels.iter().position(|l| l == &label)
    .expect("label must exist in feature set");

// BAD: bare unwrap
let idx = labels.iter().position(|l| l == &label).unwrap();
```

### Performance Patterns

- Use `hashbrown::HashMap` over `std::collections::HashMap`
- Use `smol_str::SmolStr` for strings under 22 bytes (tags, tokens)
- Pre-allocate `Vec::with_capacity(n)` when size is known
- Use `mimalloc` as global allocator (already configured)
- Avoid `clone()` in hot loops — prefer `clone_from()` or borrows
- Use `smallvec::SmallVec<[T; N]>` for collections ≤8 elements

### Large Files

Files should stay under 500 lines. If a module grows too large,
split into a directory module:

```rust
// Before: src/tokenize.rs (800 lines)
// After:
//   src/tokenize/mod.rs     — re-exports, register_module
//   src/tokenize/treebank.rs
//   src/tokenize/toktok.rs
//   src/tokenize/punkt.rs
```

## CI Pipeline

The CI workflow (`.github/workflows/quality.yml`) runs 8 jobs:

| Job | Command | Requirement |
|---|---|---|
| **fmt** | `cargo fmt --all -- --check` | Must pass (strict formatting) |
| **clippy** | `cargo clippy --all-targets -- -D correctness -D suspicious -D perf -W style -W complexity` | 0 errors. Style/complexity warnings allowed |
| **doc** | `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items` | 0 warnings |
| **test** | `cargo test --all-targets` | All tests pass |
| **audit** | `cargo audit` | No known vulnerabilities |
| **deny** | `cargo deny --all-features check licenses advisories` | License + advisory compliance |
| **hack** | `cargo hack check --feature-powerset --no-dev-deps` | All feature combinations compile |
| **bench** | `python -m benchmarks.run --ci --threshold 0.25` | No regression >25% vs baseline |

All 8 must pass for a PR to be merged.

## Release Process

### Cutting a Release

```bash
# 1. Update version in Cargo.toml, pyproject.toml, fastnltk/__init__.py
# 2. Update CHANGELOG
# 3. Run final checks
cargo test --all-targets && pytest tests/ -q
python -m benchmarks.run --save

# 4. Build + upload to PyPI
maturin build --release --out dist
twine upload dist/*.whl

# 5. Tag the release
git tag v0.x.x
git push origin v0.x.x
```

### Version Numbering

fastNLTK follows [Semantic Versioning](https://semver.org/):
- **Major**: Breaking API changes (unlikely — API matches NLTK)
- **Minor**: New features, new Rust-accelerated modules
- **Patch**: Bug fixes, performance improvements, documentation

## Getting Help

- **Issue tracker**: [github.com/your/fastnltk/issues](https://github.com/your/fastnltk/issues)
- **Discussions**: [github.com/your/fastnltk/discussions](https://github.com/your/fastnltk/discussions)
- **NLTK docs**: [nltk.org](https://www.nltk.org) — the API we mirror

For Rust-specific questions about PyO3 patterns, check:
- [PyO3 User Guide](https://pyo3.rs)
- [maturin docs](https://maturin.rs)
- [Rustlings](https://github.com/rust-lang/rustlings) — learn Rust
