# Contributing

## Setup

```bash
git clone https://github.com/wyattferguson/fastnltk && cd fastnltk
python -m venv .venv && source .venv/bin/activate
pip install maturin && maturin develop --release
pip install -e ".[dev,test,lint]"
python -m nltk.downloader punkt averaged_perceptron_tagger wordnet
```

## Rules

1. **Python shims must delegate to Rust** — no pure-Python reimplementations.
   If a feature stays in Python, document why in the module docstring.
2. **No unsafe code** — `#![deny(unsafe_code)]` is set at the crate level.
3. **`unsafe_code = "deny"`** — no exceptions.
4. **Conventional commits** — `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`.
5. **Run lints before pushing**:
   ```bash
   cargo clippy --all-targets -- -D warnings
   cargo fmt --all -- --check
   ruff check fastnltk/ tests/
   cargo test
   python -m pytest
   ```

## CI

Pushes to `main` run full lint + test matrix. Tagged `v*` pushes trigger
PyPI release. See `.github/workflows/`.

## Project structure

- `src/` — Rust crate (`fastnltk._rust`), mirrors NLTK module layout
- `fastnltk/` — Python shims, re-exports from Rust + delegates to `nltk`
- `tests/` — pytest suite, mirrors `fastnltk/` layout
- `benchmarks/` — Custom harness for regression-tracked comparisons against `nltk`
