# fastNLTK — Exhaustive Implementation Plan

A drop-in Rust-accelerated replacement for NLTK. Same API, same behavior, 5-50x faster on hot paths.

---

## 0. Guiding Principles

| Principle | Rationale |
|---|---|
| **API-identical** | `from fastnltk import word_tokenize` works exactly like `from nltk.tokenize import word_tokenize`. Same fn signatures, same defaults, same return types. |
| **Delegation over rewrite** | Pure-Python wrappers for non-hot modules (corpus readers, GUI, inference). Only rewrite the 20% of code that runs 80% of the time. |
| **Rust on hot paths only** | Tokenization, stemming, tagging, LM, collocations, classification, string metrics get Rust impls. Everything else wraps original NLTK or is thin Python. |
| **Progressive disclosure** | Ship working stub first, then replace module by module. Each module independently testable against NLTK's test suite. |
| **Zero-copy where possible** | Rust string processing avoids Python allocations until final result. Use `&str` slices over `Cow`, yield Python strings only at boundary. |
| **Corpus data compatible** | Use existing `nltk_data` directory. Do not invent new data formats or require re-downloads. |
| **abi3 for wheels** | Target `abi3-py38` so one wheel covers Python 3.8+. |

---

## 1. Project Scaffold & Build System

### 1.1 Directory Layout

```
fastnltk/
├── Cargo.toml                  # Rust crate — cdylib + rlib
├── pyproject.toml              # maturin build config
├── Makefile                    # dev/build/lint/test targets
├── rust-toolchain.toml         # MSRV pinning
├── .github/
│   └── workflows/
│       ├── ci.yml              # test matrix: 3.8-3.13, pypy, ubuntu/macos/windows
│       ├── release.yml         # build + publish wheels to PyPI
│       └── benchmark.yml       # nightly perf tracking
├── fastnltk/                   # Python package (shim layer) — `python-source` in pyproject.toml
│   ├── __init__.py             # Re-exports everything from submodules
│   ├── _rust.pyi               # Type stubs for compiled Rust extension module
│   ├── _rust/                  # (maturin places compiled .pyd/.so under this namespace)
│   ├── tokenize.py             # Python shim → Rust or fallback to nltk
│   ├── tag.py
│   ├── stem.py
│   ├── classify.py
│   ├── collocations.py
│   ├── probability.py
│   ├── lm.py
│   ├── metrics.py
│   ├── chunk.py
│   ├── parse.py                # Pure Python shim (no Rust)
│   ├── tree.py                 # Pure Python shim
│   ├── corpus/                 # Pure Python wrappers for nltk_data
│   ├── sem.py
│   ├── translate.py
│   ├── sentiment.py
│   ├── data.py                 # nltk.data equivalent — find nltk_data dir, load pickles
│   ├── inference.py
│   ├── cluster.py
│   ├── ccg.py
│   ├── chat.py
│   └── downloader.py           # Wrap nltk.downloader
├── src/                        # Rust source
│   ├── lib.rs                  # Crate root — PyO3 module registration
│   ├── prelude.rs              # Common imports
│   ├── data.rs                 # NLTK data loader (pickle → bincode converter)
│   ├── tokenize/
│   │   ├── mod.rs
│   │   ├── punkt.rs            # Punkt sentence tokenizer (port from NLTK)
│   │   ├── regexp.rs           # Regex-based tokenizers
│   │   ├── treebank.rs         # Treebank tokenizer
│   │   ├── tweet.rs            # TweetTokenizer
│   │   └── simple.rs           # Line/Space/Tab tokenizers
│   ├── stem/
│   │   ├── mod.rs
│   │   ├── rustling_stemmers.rs  # Wrapper around rust-stemmers crate
│   │   ├── porter.rs           # Porter stemmer (not in rust-stemmers)
│   │   ├── lancaster.rs        # Lancaster stemmer
│   │   ├── wordnet.rs          # WordNet lemmatizer (morphy algorithm)
│   │   ├── isri.rs             # Arabic ISRI stemmer
│   │   ├── cistem.rs           # German Cistem
│   │   ├── rslp.rs             # Portuguese RSLP stemmer
│   │   └── arlstem.rs          # Arabic ARLSTem
│   ├── tag/
│   │   ├── mod.rs
│   │   ├── perceptron.rs       # Wrapper around rustling perceptron_pos_tagger
│   │   ├── tnt.rs              # TnT tagger (trigram HMM + Viterbi)
│   │   ├── sequential.rs       # Ngram/Uni/Bi/Trigram taggers
│   │   └── hmm.rs              # Wrapper around rustling HMM
│   ├── classify/
│   │   ├── mod.rs
│   │   ├── naivebayes.rs       # NaiveBayesClassifier training + prediction
│   │   └── maxent.rs           # MaxentClassifier GIS/IIS training
│   ├── collocations.rs         # Bigram/Trigram/Quadgram finder
│   ├── probability.rs          # FreqDist, ConditionalFreqDist, ProbDist types
│   ├── lm.rs                   # Language model wrappers around rustling LM
│   ├── metrics/
│   │   ├── mod.rs
│   │   ├── distance.rs         # edit_distance, jaro, jaro_winkler, dice (ported from vtext)
│   │   ├── association.rs      # BigramAssocMeasures, TrigramAssocMeasures
│   │   ├── scores.rs           # precision, recall, f_measure
│   │   └── segmentation.rs     # windowdiff, pk, bcubed
│   ├── chunk.rs               # RegexpChunkParser — regex grammar compilation + matching
│   └── util/
│       ├── mod.rs
│       ├── regex_cache.rs      # LRU regex compilation cache
│       └── string.rs           # String utility functions
├── tests/                      # Python integration tests
│   ├── test_tokenize.py
│   ├── test_stem.py
│   ├── test_tag.py
│   ├── test_classify.py
│   ├── test_collocations.py
│   ├── test_probability.py
│   ├── test_lm.py
│   ├── test_metrics.py
│   └── conftest.py
├── benchmarks/                 # pytest-benchmark comparisons vs NLTK
│   ├── tokenize_bench.py
│   ├── stem_bench.py
│   ├── tag_bench.py
│   ├── collocations_bench.py
│   └── full_pipeline_bench.py
└── scripts/
    ├── convert_models.py       # Convert NLTK pickles → Rust bincode
    └── run_nltk_tests.py       # Run NLTK's test suite against fastNLTK
```

### 1.2 Cargo.toml

```toml
[package]
name = "fastnltk"
version = "0.1.0"
edition = "2021"
license = "Apache-2.0"
homepage = "https://github.com/your/fastnltk"
repository = "https://github.com/your/fastnltk"
rust-version = "1.75"     # MSRV — match PyO3's minimum

[lib]
name = "fastnltk"
crate-type = ["cdylib", "rlib"]   # rlib allows Rust benchmarks + tests

[dependencies]
# ── Core bindings ──────────────────────────────────────
pyo3 = { version = "0.23", features = ["abi3-py38", "extension-module"] }

# ── Tokenization ───────────────────────────────────────
regex = "1"
unicode-segmentation = "1"

# ── Stemming (saves 5,900 LoC) ─────────────────────────
rust-stemmers = "1"

# ── POS tagging + LM + HMM + ngram (saves ~2,500 LoC) ──
rustling = { version = "0.8", default-features = false, features = ["parallel"] }

# ── Serialization ──────────────────────────────────────
serde = { version = "1", features = ["derive"] }
bincode = "2"

# ── Performance ────────────────────────────────────────
hashbrown = "0.15"
rustc-hash = "2"
once_cell = "1"
parking_lot = "0.12"

# ── Optional features ──────────────────────────────────
rayon = { version = "1", optional = true }
whatlang = { version = "0.16", optional = true }

[features]
default = ["parallel"]
parallel = ["rayon", "rustling/parallel"]
language-detection = ["whatlang"]

[profile.release]
lto = "fat"               # maximally optimizes across crate boundaries
codegen-units = 1         # single compilation unit = better optimization
strip = true              # remove debug symbols from wheel
opt-level = 3
panic = "unwind"

[profile.dev]
opt-level = 0             # fast compilation in dev

[dev-dependencies]
approx = "0.5"            # float comparison in Rust tests
tempfile = "3"
```

### 1.3 pyproject.toml

```toml
[build-system]
requires = ["maturin>=1.7,<2.0"]
build-backend = "maturin"

[project]
name = "fastnltk"
version = "0.1.0"
requires-python = ">=3.8"
description = "Drop-in Rust-accelerated replacement for NLTK"
authors = [
    { name = "Your Name", email = "your@email.com" },
]
license = "Apache-2.0"
classifiers = [
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: Implementation :: CPython",
    "Programming Language :: Python :: Implementation :: PyPy",
    "Programming Language :: Rust",
    "Topic :: Text Processing :: Linguistic",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
]
dependencies = []
dynamic = ["readme"]

# ── Optional dependencies ─────────────────────────────────
[project.optional-dependencies]
dev = ["maturin", "pytest", "pytest-benchmark", "ruff", "mypy"]
test = ["pytest", "pytest-benchmark", "hypothesis", "nltk"]
lint = ["ruff", "mypy", "pre-commit"]
all = ["fastnltk[dev,test,lint]"]

[tool.maturin]
python-source = "fastnltk"
module-name = "fastnltk._rust"
bindings = "pyo3"
features = ["pyo3/extension-module", "abi3-py38"]

[tool.ruff]
line-length = 100
target-version = "py38"

[tool.ruff.lint]
extend-select = ["I", "Q", "UP", "RUF100"]
extend-ignore = ["E501"]   # line-length handled by formatter

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
filterwarnings = ["error"]
timeout = 60
```

### 1.4 Makefile (patterns from pydantic-core)

For consistency, adopt the Makefile pattern used by professional PyO3 projects:

```makefile
.DEFAULT_GOAL := all
sources = fastnltk tests

export CARGO_TERM_COLOR = $(shell (test -t 0 && echo "always") || echo "auto")

.PHONY: install
install:  ## Install deps + pre-commit hooks
	uv sync --group all
	uv run pre-commit install --install-hooks

.PHONY: build-dev
build-dev:  ## Build fast dev version
	uv run maturin develop --uv

.PHONY: build-prod
build-prod:  ## Build optimized release
	uv run maturin develop --uv --release

.PHONY: format
format:  ## Auto-format Rust + Python
	uv run ruff check --fix $(sources)
	uv run ruff format $(sources)
	cargo fmt

.PHONY: lint-python
lint-python:  ## Lint Python source
	uv run ruff check $(sources)
	uv run ruff format --check $(sources)

.PHONY: lint-rust
lint-rust:  ## Lint Rust source
	cargo fmt --all -- --check
	cargo clippy --tests -- -D warnings

.PHONY: lint
lint: lint-python lint-rust  ## Lint all

.PHONY: test
test:  ## Run Python tests
	uv run pytest

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
	rm -f fastnltk/_rust/*.so fastnltk/_rust/*.dll fastnltk/_rust/*.dylib

.PHONY: all
all: format build-dev lint test  ## CI standard check set
```

### 1.5 rust-toolchain.toml

```toml
[toolchain]
channel = "stable"
components = ["rustfmt", "clippy"]
```

### 1.6 Pre-commit hooks (.pre-commit-config.yaml)

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.9.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/doublify/pre-commit-rust
    rev: v1.0
    hooks:
      - id: fmt
      - id: clippy
```

---

## 2. Architecture: Python Shim → Rust Core

### 2.1 The Shim Pattern (inspired by pydantic-core)

The module naming follows the pydantic-core pattern:
- Compiled Rust module: `fastnltk._rust` (a PyO3 extension module)
- Python shim: `fastnltk/__init__.py` re-exports from `fastnltk._rust`
- Python source lives in `fastnltk/` (configured via `python-source` in pyproject.toml)

```
User code:  from fastnltk import word_tokenize
                    │
            ┌───────▼───────┐
Layer 1:    │  __init__.py   │  Re-exports, possibly wraps for compat
            │  (pure Python) │
            └───────┬───────┘
                    │
            ┌───────▼───────┐
Layer 2:    │  tokenize.py   │  Python shim: validates args, calls Rust
            │  (thin shim)   │  Falls back to pure-Python NLTK if Rust
            └───────┬───────┘  unimpl or data missing
                    │
            ┌───────▼───────┐
Layer 3:    │  _rust         │  Rust PyO3 extension module
            │  (compiled)    │  Actual computation
            └───────────────┘
```

**Shim responsibilities**:
- Accept Python types, validate, call Rust via `fastnltk._rust.fn(...)`
- Convert Rust return values back to Python types (list of str, tuple of (str, tag), etc.)
- Provide fallback to original NLTK when a function is not yet Rust-ported
- Handle `nltk_data` path resolution exactly as NLTK does

### 2.2 Type Stubs (`_rust.pyi`)

Ship a `.pyi` stub file so IDE autocomplete works for Rust-exported functions:

```python
# fastnltk/_rust.pyi

def sent_tokenize(text: str, language: str = "english") -> list[str]: ...
def word_tokenize(text: str, language: str = "english", preserve_line: bool = False) -> list[str]: ...

class PunktSentenceTokenizer:
    def __init__(self, train_text: str | None = None, language: str = "english") -> None: ...
    def tokenize(self, text: str) -> list[str]: ...
    def span_tokenize(self, text: str) -> list[tuple[int, int]]: ...
    def sentences_from_text(self, text: str) -> list[str]: ...

class SnowballStemmer:
    def __init__(self, language: str = "english") -> None: ...
    def stem(self, word: str) -> str: ...
    def stem_many(self, words: list[str]) -> list[str]: ...

class PerceptronTagger:
    def __init__(self) -> None: ...
    def tag(self, tokens: list[str]) -> list[tuple[str, str]]: ...
    def tag_sents(self, sentences: list[list[str]]) -> list[list[tuple[str, str]]]: ...
```

**Verify stubs at CI time** with `mypy.stubtest` (pattern from pydantic-core):
```bash
uv run python -m mypy.stubtest fastnltk._rust --allowlist .mypy-stubtest-allowlist
```

### 2.3 Data Layer Compatibility

NLTK stores data in `~/nltk_data` (or `NLTK_DATA` env var). fastNLTK must:

1. **Read** serialized models (Punkt pickle, tagger pickles, tokenizer data) from existing `nltk_data/`.
2. **Not require re-download** of any data.
3. **Support** `nltk.download('punkt')` → fastNLTK uses same downloaded data.
4. **For Rust-native models**: store them in bincode format in same directory structure.

**Data file resolution** (port of `nltk.data.find`):
```
Search order:
  1. $NLTK_DATA env var
  2. ~/nltk_data/
  3. C:/nltk_data/  (Windows)
  4. /usr/share/nltk_data/
  5. /usr/local/share/nltk_data/
  6. Via nltk package data
```

### 2.4 Model Conversion Pipeline

NLTK stores models as Python pickles (e.g., `tokenizers/punkt/english.pickle`).
fastNLTK needs to load them into Rust structs.

**Strategy** (two-phase):
1. **Phase 1 (MVP)**: Load pickle in Python shim, pass preprocessed data to Rust.
2. **Phase 2 (optimized)**: Ship a `scripts/convert_models.py` that converts pickles → bincode.
   Run once during `post-install` or on first `import`.

```python
# scripts/convert_models.py
"""Convert NLTK pickle models to fastNLTK bincode format."""
import pickle
import nltk.data

from fastnltk._rust import convert_punkt_model, convert_perceptron_model

# Convert Punkt tokenizer
punkt_path = nltk.data.find('tokenizers/punkt/english.pickle')
with open(punkt_path, 'rb') as f:
    model = pickle.load(f)
convert_punkt_model(model, 'fastnltk/data/punkt_english.bin')

# Convert Averaged Perceptron tagger
ap_path = nltk.data.find('taggers/averaged_perceptron_tagger/')
# Load pickle weights, convert, save as bincode
```

### 2.5 Error Handling

| Python Exception | Rust Counterpart | When |
|---|---|---|
| `ValueError` | `PyValueError` | Invalid arg, unknown language |
| `LookupError` | `PyLookupError` | Missing resource file |
| `TypeError` | `PyTypeError` | Wrong arg type |
| `StopIteration` | `PyStopIteration` | Iterator exhaustion |

Every Rust function returns `PyResult<T>`. Internal Rust errors are converted via:
```rust
.map_err(|e| PyValueError::new_err(e.to_string()))
```

### 2.6 GIL Management

- Release GIL during CPU-bound computation: `Python::allow_threads(|| { ... })`
- Tokenization of large texts: yield tokens one-at-a-time via Python generators
- Model loading: use `OnceLock` + `parking_lot::RwLock` for thread-safe lazy caching

---

## 3. Existing Rust Crates We Depend On

### 3.1 Direct Dependencies (add to Cargo.toml)

These crates are mature, correct, and map directly onto NLTK features.
We wrap them with thin PyO3 adapters.

| Crate | What It Gives Us | NLTK LoC Replaced | Our Wrapper LoC | License |
|---|---|---|---|---|
| `rust-stemmers` v1.0 | All 16 Snowball stemmers (Da, Du, En, Fi, Fr, De, Hu, It, No, Po, Ro, Ru, Es, Sv, Tr, Ar) | 5,921 LoC (`nltk.stem.snowball`) | ~80 | MIT |
| `rustling` v0.8 | Averaged perceptron POS tagger, LM (MLE/Lidstone/Laplace/KneserNey), HMM, ngram + Viterbi | ~2,500 LoC (`nltk.tag.perceptron`, `nltk.lm.*`, `nltk.tag.hmm`) | ~300 | MIT |
| `unicode-segmentation` v1.12 | Unicode word/sentence boundaries per TR#29 | ~300 LoC (manual Unicode impl) | ~10 | MIT/Apache-2.0 |
| `regex` v1 | Guaranteed linear-time DFA regex engine | Infinite (can't write our own) | — | MIT/Apache-2.0 |
| `hashbrown` v0.15 | Faster HashMaps for FreqDist, tagger weights | — | — | MIT/Apache-2.0 |
| `rustc-hash` v2.1 | FxHashMap for small-key maps (proven by rustling) | — | — | Apache-2.0/MIT |

### 3.2 Code Ported from Crates (not direct deps)

These crates are too old or wrong-license to depend on, but their algorithms
are correct and licensed compatibly. We copy/adapt the source code.

| Source | What We Port | LoC | License | Why Not Dep |
|---|---|---|---|---|
| `vtext` tokenize | English contraction rules, VTextTokenizer | ~300 | Apache-2.0 | v0.2, unmaintained since 2021, uses ancient PyO3 0.10 |
| `vtext` metrics | `edit_distance`, `jaro_similarity`, `jaro_winkler_similarity`, `dice_similarity` | ~300 | Apache-2.0 | Same reason |
| `nltk_rs` (plutonium-guy) | Regex flag mapping (Python flags → Rust flags), parallel batch patterns | 0 (fresh impl) | LGPL-3.0 | LGPL incompatible with Apache-2.0 — **study only** |

### 3.3 What We Still Write from Scratch

Despite using all the above crates, we still write ~8,500 LoC of Rust for NLTK-specific code:

| Module | Why | Est LoC | Status |
|---|---|---|---|
| Punkt sentence tokenizer | NLTK's trained model format. No existing Rust crate does this. Port algorithm + load pickle. | ~1,200 | ✅ Complete |
| Treebank tokenizer | Contraction rules, PTB conventions. NLTK-specific regex rules. | ~400 | ✅ Complete |
| Tweet tokenizer | Emoji, hashtag, URL regex patterns. NLTK-specific implementation. | ~300 | ✅ Complete |
| RegexpTokenizer | Simple but NLTK-specific gap/match mode convention | ~150 | ✅ Complete |
| Simple tokenizers | LineTokenizer, SpaceTokenizer, TabTokenizer | ~100 | ✅ Complete |
| Porter stemmer | Not in rust-stemmers (only Snowball). Port the algorithm. | ~200 | ✅ Complete |
| Lancaster stemmer | Not in rust-stemmers. Port. | ~200 | ✅ Complete |
| WordNet lemmatizer | Morphy algorithm + exception rules from WordNet data | ~300 | ✅ Complete |
| Other stemmers | ISRI (Arabic), Cistem (German), RSLP (Portuguese) | ~800 | ✅ Complete |
| FreqDist + ProbDists | Counter-like with NLTK methods wrapping hashbrown::HashMap | ~500 | ✅ Complete |
| Collocation finders | Bigram/Trigram/Quadgram + association scoring | ~500 | ✅ Complete |
| MaxEnt classifier | GIS training loop | ~600 | ✅ Complete |
| NaiveBayes classifier | Training + prediction with Laplace smoothing | ~300 | ✅ Complete |
| TnT tagger | Trigram HMM + Viterbi + backoff smoothing | ~400 | ✅ Complete |
| Language model bridge | Wrap rustling LM (MLE, Lidstone, Laplace) | ~400 | ✅ Complete |
| VADER sentiment | Rule-based sentiment intensity | ~200 | ✅ Complete |
| BLEU/corpus BLEU | Translation scoring | ~150 | ✅ Complete |
| RegexpChunkParser | Grammar compilation + tag matching (NLTK's chunk.regexp) | ~300 | ✅ Complete |
| Data layer | `nltk_data` finder, pickle loader, bincode converter | ~300 | ✅ Complete |
| TextCat bridge | whatlang wrapper | ~50 | ✅ Complete |
| ARLSTem / ARLSTem2 | Arabic stemmer variants | ~200 | 📋 Pending |
| MWETokenizer | Multi-word expression matching | ~200 | 📋 Pending |
| Sequential taggers | Ngram/Uni/Bi/Trigram/Default/Affix/RegexpTagger | ~500 | 📋 Pending |
| HMM tagger | HiddenMarkovModelTagger wrapper | ~200 | 📋 Pending |
| String metrics | Association measures, segmentation | ~400 | 📋 Pending |
| **Total** | | **~9,500 LoC** | **~80% Complete** |

---

## 4. Module-by-Module Implementation Order

### Phase 0 — Foundation (Week 1-2) ✅ Complete

| Task | Output | Status |
|---|---|---|
| Scaffold project with maturin | Working `fastnltk._rust` importable from Python | ✅ |
| `fastnltk/__init__.py` | Module tree that re-exports all submodules | ✅ |
| `fastnltk/data.py` | Data file resolution (wraps `nltk.data` + own fallback) | ✅ |
| `fastnltk/_rust.pyi` | Type stubs for Rust exports | ✅ |
| `src/lib.rs` | Module registration with all submodules | ✅ |
| `src/data.rs` | NLTK data file loading + pickle → bincode | ✅ |
| `src/util/mod.rs` | Regex cache, string utils | ✅ |
| Python shims (all modules) | Each shim delegates to `_rust` or falls back to NLTK | ✅ |
| Makefile | `build-dev`, `build-prod`, `lint`, `test` targets | ✅ |
| CI pipeline | GitHub Actions: build + test on 3.8-3.13, ubuntu/macos/windows | 📋 Planned |
| Benchmark harness | `pytest-benchmark` comparison runs against NLTK | 📋 Planned |

### Phase 1 — Tokenization (Week 3-4) ✅ Complete

**Rust modules to implement**:

| Module | File | Key Types/Fns | Est Speedup | Status |
|---|---|---|---|---|
| Regex tokenizers | `src/tokenize/regexp.rs` | `RegexpTokenizer`, `WhitespaceTokenizer`, `WordPunctTokenizer`, `BlanklineTokenizer` | 10-30x | ✅ |
| Simple tokenizers | `src/tokenize/simple.rs` | `LineTokenizer`, `SpaceTokenizer`, `TabTokenizer`, `CharTokenizer` | 5-10x | ✅ |
| Treebank tokenizer | `src/tokenize/treebank.rs` | `TreebankWordTokenizer`, `TreebankWordDetokenizer` | 10-20x | ✅ |
| Tweet tokenizer | `src/tokenize/tweet.rs` | `TweetTokenizer` — emoji, hashtag, URL regex | 10-20x | ✅ |
| Punkt tokenizer | `src/tokenize/punkt.rs` | `PunktSentenceTokenizer`, `PunktTokenizer` | 10-50x | ✅ |
| TokTok tokenizer | `src/tokenize/toktok.rs` | `ToktokTokenizer` | 10-20x | 📋 Pending |
| MWE tokenizer | `src/tokenize/mwe.rs` | `MWETokenizer` | 5-10x | 📋 Pending |
| NIST tokenizer | `src/tokenize/nist.rs` | `NISTTokenizer` | 5-10x | 📋 Pending |
| SExpr tokenizer | `src/tokenize/sexpr.rs` | `SExprTokenizer`, `sexpr_tokenize` | 5-10x | 📋 Pending |
| TextTiling | `src/tokenize/texttiling.rs` | `TextTilingTokenizer` | 3-8x | 📋 Pending |

**Key techniques**:
- Compile regexes once with `once_cell::sync::Lazy` or LRU cache
- Return `Vec<String>` for batch or yield via Python generator for streaming
- Punkt: port entire sentence detector, load existing trained pickle data
- `sent_tokenize()` + `word_tokenize()` — the two most-called NLTK functions — must be fastest

### Phase 2 — Stemming (Week 5) ✅ Complete

| Module | File | Key Types | Est Speedup | Status |
|---|---|---|---|---|
| Snowball stemmer | `src/stem/snowball.rs` | `SnowballStemmer` — wraps `rust-stemmers` crate | 15-20x | ✅ |
| Porter stemmer | `src/stem/porter.rs` | `PorterStemmer` — port the algorithm | 10-20x | ✅ |
| Lancaster stemmer | `src/stem/lancaster.rs` | `LancasterStemmer` | 10-20x | ✅ |
| ISRI stemmer | `src/stem/isri.rs` | `ISRIStemmer` (Arabic) | 10-15x | ✅ |
| Cistem | `src/stem/cistem.rs` | `Cistem` (German) | 10-15x | ✅ |
| RSLP stemmer | `src/stem/rslp.rs` | `RSLPStemmer` (Portuguese) | 5-10x | ✅ |
| Regexp stemmer | `src/stem/regexp.rs` | `RegexpStemmer` | 10-20x | ✅ |
| WordNet lemmatizer | `src/stem/wordnet.rs` | `WordNetLemmatizer` (morphy fn) | 5-10x | ✅ |
| ARLSTem / ARLSTem2 | `src/stem/arlstem.rs` | Arabic stemmer variants | 10-15x | 📋 Pending |

**Key techniques**:
- `rust-stemmers` covers 16 Snowball languages. Exception lists (English irregulars, etc.) are data tables, not algorithm — include at compile time.
- WordNet lemmatizer: load data from `nltk_data/corpora/wordnet/`, port morphy algorithm

### Phase 3 — POS Tagging (Week 6-7) ✅ Complete

| Module | File | Key Types | Est Speedup | Status |
|---|---|---|---|---|
| Perceptron tagger | `src/tag/perceptron.rs` | `PerceptronTagger` — wraps `rustling::perceptron_pos_tagger` | 5-6x | ✅ |
| TnT tagger | `src/tag/tnt.rs` | `TnT` — trigram HMM + Viterbi | 5-10x | ✅ |
| Sequential taggers | pending | `DefaultTagger`, `NgramTagger`, `UnigramTagger`, `BigramTagger`, `TrigramTagger`, `AffixTagger`, `RegexpTagger` | 3-10x | 📋 Pending |
| HMM tagger | pending | `HiddenMarkovModelTagger` — wraps `rustling::hmm` | 5-10x | 📋 Pending |

**Key techniques**:
- Load PerceptronTagger weights from NLTK pickle (`taggers/averaged_perceptron_tagger/`), convert to rustling's FlatBuffers/gzip-JSON format on first load
- TnT: load from `taggers/tnt/` — port smoothing + Viterbi to Rust
- Feature extraction (prefix, suffix, shape, prev tag) in tight loops → compiled code
- Model caching: load once, store in `OnceLock<RwLock<HashMap<String, Model>>>`

### Phase 4 — Classification (Week 8) ✅ Complete

| Module | File | Key Types | Est Speedup | Status |
|---|---|---|---|---|
| Naive Bayes | `src/classify/naivebayes.rs` | `NaiveBayesClassifier` — train + classify | 3-5x | ✅ |
| MaxEnt | `src/classify/maxent.rs` | `MaxentClassifier` — GIS training | 3-8x | ✅ |
| Positive Naive Bayes | `src/classify/naivebayes.rs` | `PositiveNaiveBayesClassifier` | 3-5x | ✅ |
| TextCat | `src/classify/textcat.rs` | `TextCat` via `whatlang` dep | 10-50x | ✅ |
| Decision Tree | shim (tiny) | `DecisionTreeClassifier` | 1x | ✅ (shim) |

**Key techniques**:
- Training loop releases GIL
- Sparse feature vectors: `Vec<(usize, f64)>` instead of Python dicts
- MaxEnt GIS/IIS iterative scaling — convergence loops benefit massively from compiled code

### Phase 5 — Collocations & Probability (Week 9) ✅ Complete

| Module | File | Key Types | Est Speedup | Status |
|---|---|---|---|---|
| Collocations | `src/collocations.rs` | `BigramCollocationFinder`, `TrigramCollocationFinder`, `QuadgramCollocationFinder` | 5-15x | ✅ |
| FreqDist | `src/probability.rs` | `FreqDist`, `ConditionalFreqDist` | 5-10x | ✅ |

**Key techniques**:
- FreqDist: reimplemented with `hashbrown::HashMap` + custom methods matching NLTK's API
- Collocation: count ngram frequencies, score with association measures (PMI, chi-square, log-likelihood)
- Association measures: precompute in Rust, single pass

### Phase 6 — Language Models (Week 10) ✅ Complete

| Module | File | Key Types | Est Speedup | Status |
|---|---|---|---|---|
| Ngram LM | `src/lm.rs` | Wraps `rustling::lm::{MLE, Lidstone, Laplace}` | 10-39x | ✅ |

**Key techniques**:
- Proven by rustling: 11x fitting, 25-39x generation, 2x scoring
- Vocabulary: `hashbrown::HashMap<String, usize>` for O(1) lookup

### Phase 7 — String Metrics (Week 10-11) ✅ Complete

| Module | File | Key Types | Est Speedup | Status |
|---|---|---|---|---|
| Distance | `src/metrics/jaro.rs`, `src/metrics/jaccard.rs` | `edit_distance`, `jaccard_distance`, `binary_distance`, `masi_distance`, `jaro`, `jaro_winkler`, `dice` | 5-50x | ✅ |
| Scores | `src/metrics/scores.rs` | `precision`, `recall`, `f_measure` | 2-5x | ✅ |
| Segmentation/BLEU | shim | `windowdiff`, `pk`, `bcubed`, `association` measures | 📋 Pending |

### Phase 8 — Chunking (Week 11) ✅ Complete

| Module | Strategy | Est Speedup | Status |
|---|---|---|---|
| `RegexpParser` | Port chunk grammar compiler + tag sequence matcher to Rust | 5-10x | ✅ |
| `NEChunkParser` | Shim — works via MaxEnt perceptron (Python shim) | 1x | ✅ (shim) |
| Chunk util | Shim (tree conversion functions) | 1x | ✅ (shim) |

### Phase 9 — Python Shim Completeness (Next)

All 22 module files exist. Remaining work is to fill gaps in re-exports:

#### 9.1 Pure-Python shims (no Rust, re-export from NLTK)

| File | Status | Missing |
|---|---|---|
| `fastnltk/ccg.py` | ✅ | — uses `from nltk.ccg import *` |
| `fastnltk/chat.py` | ✅ | — uses `from nltk.chat import *` |
| `fastnltk/cluster.py` | ✅ | — uses `from nltk.cluster import *` |
| `fastnltk/corpus/__init__.py` | ✅ | — uses `from nltk.corpus import *` |
| `fastnltk/data.py` | ✅ | — find, load, path, show_cfg, data_dirs |
| `fastnltk/downloader.py` | ✅ | — download, download_shell, download_gui, update |
| `fastnltk/inference.py` | ✅ | — uses `from nltk.inference import *` |
| `fastnltk/parse.py` | ✅ | — uses `from nltk.parse import *` |
| `fastnltk/sem.py` | ✅ | — uses `from nltk.sem import *` |
| `fastnltk/tree.py` | ✅ | — uses `from nltk.tree import *` |

#### 9.2 Rust-backed modules — missing Python re-exports

For each Rust-backed module, `from fastnltk.X import Y` must work for every public `Y` in `nltk.X`.
Currently missing symbols by module (measured against `nltk-3.9`):

| Module | Rust classes | Missing submodule re-exports | Missing fns/classes | Priority |
|---|---|---|---|---|
| `tokenize` | 13 classes + 2 fns | punkt, treebank, regexp, simple, repp, toktok, mwe, sexpr, texttiling, stanford_segmenter, casual, destructive, api, util, load | blankline_tokenize, line_tokenize, sexpr_tokenize, wordpunct_tokenize, NLTKWordTokenizer | **P0** — most-used module |
| `stem` | 7 classes + WordNetLemmatizer | snowball, porter, lancaster, isri, cistem, rslp, regexp, wordnet, arlstem, api, util | StemmerI, ARLSTem, ARLSTem2 | **P1** |
| `tag` | 2 classes (Perceptron, TnT) | perceptron, tnt, sequential, brill, crf, hmm, hunpos, senna, stanford, api, mapping, util | PRETRAINED_TAGGERS | **P1** |
| `classify` | 4 classes (NB, PositiveNB, MaxEnt, TextCat) | naivebayes, maxent, textcat, decisiontree, megam, scikitlearn, senna, weka, api, util | ConditionalExponentialClassifier, RTEFeatureExtractor, WekaClassifier | **P2** |
| `collocations` | 3 classes (Bi/Tri/Quadgram) | — | AbstractCollocationFinder, BigramAssocMeasures, TrigramAssocMeasures, QuadgramAssocMeasures, ContingencyMeasures | **P1** |
| `probability` | 2 classes (FreqDist, CondFreqDist) | — | DictionaryConditionalProbDist, RandomProbDist + internal imports | **P2** |
| `lm` | 3 classes (MLE, Lidstone, Laplace) | counter, models, preprocessing, smoothing, util, vocabulary, api | KneserNeyInterpolated, AbsoluteDiscountingInterpolated, WittenBellInterpolated, StupidBackoff, Vocabulary, NgramCounter | **P2** |
| `metrics` | 9 fns | — | alignment_error_rate | **P3** (rarely used) |
| `sentiment` | 1 class (VADER) | vader, sentiment_analyzer, api | — | **P2** |
| `chunk` | 1 class (RegexpParser) | regexp, ne_chunker, named_entity, api, util | — | **P2** |
| `translate` | 2 fns (BLEU) | — | IBM models, stack_decoder, phrase_based (shim only) | **P3** |

#### 9.3 Shim Completeness Checklist (Phase 9)

For each Rust-backed module, ensure:

```python
# Pattern for missing submodule re-exports (e.g., in fastnltk/tokenize.py)
from nltk.tokenize import punkt as _nltk_punkt
punkt = _nltk_punkt

# Pattern for missing class re-exports (simple pass-through)
# Add to existing from-import or:
from nltk.tokenize import NLTKWordTokenizer

# Pattern for missing convenience functions
# Already exist in nltk, just re-export:
from nltk.tokenize import blankline_tokenize, line_tokenize, wordpunct_tokenize
```

The goal: `set(dir(nltk.tokenize)) - set(dir(fastnltk.tokenize))` should be empty
for all 11 Rust-backed modules (excluding `__pycache__`, private attrs).

Verification script:
```python
import nltk, fastnltk
modules = ['tokenize', 'tag', 'stem', 'classify', 'collocations',
           'probability', 'lm', 'metrics', 'sentiment', 'translate', 'chunk']
for m in modules:
    n = set(dir(getattr(nltk, m)))
    f = set(dir(getattr(fastnltk, m)))
    missing = {x for x in n if not x.startswith('_')} - {x for x in f if not x.startswith('_')}
    if missing:
        print(f'{m}: {len(missing)} missing — {sorted(missing)[:5]}...')
```

#### 9.4 Effort Estimate

| Task | Files | Est lines | Est time |
|---|---|---|---|
| Submodule re-exports (`punkt = _nltk_punkt`) | 7 files | ~80 | 1 day |
| Missing class re-exports (add to `from nltk.X import Y`) | 5 files | ~30 | 0.5 day |
| Missing fn re-exports (`blankline_tokenize`, etc.) | 3 files | ~15 | 0.5 day |
| Association measures shim (BigramAssocMeasures, etc.) | 1 file | ~20 | 0.5 day |
| ARLSTem/ARLSTem2 Python shim + Rust min-port | 2 files | ~150 | 1 day |
| Sequential taggers Python shim | 1 file | ~30 | 0.5 day |
| Verification + test | — | — | 0.5 day |
| **Total** | | **~325** | **~4 days** |

---

## 5. Rust Module Implementation Patterns

### 5.1 PyO3 Function Template (with GIL release)

```rust
use pyo3::prelude::*;

/// Tokenize a single text into sentences using Punkt.
#[pyfunction(signature = (text, lang="english"))]
#[pyo3(name = "sent_tokenize")]
fn sent_tokenize_py(text: &str, lang: &str) -> PyResult<Vec<String>> {
    py.allow_threads(|| {
        let tokenizer = get_punkt_tokenizer(lang)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let sentences: Vec<String> = tokenizer
            .sentences_from_text(text)
            .into_iter()
            .map(String::from)
            .collect();
        Ok(sentences)
    })
}
```

### 5.2 PyO3 Class Template (with Lazy model loading)

```rust
#[pyclass(name = "SnowballStemmer", module = "fastnltk._rust")]
struct SnowballStemmer {
    inner: stemmer::Algorithm,
}

#[pymethods]
impl SnowballStemmer {
    #[new]
    #[pyo3(signature = (language="english"))]
    fn new(language: &str) -> PyResult<Self> {
        let algo = language_to_algorithm(language)
            .ok_or_else(|| PyValueError::new_err(format!("Unknown language: {language}")))?;
        Ok(Self { inner: algo })
    }

    fn stem(&self, word: &str) -> String {
        let mut stemmer = stemmer::Stemmer::create(self.inner);
        stemmer.stem(word).to_string()
    }

    /// Batch stem for speed
    fn stem_many(&self, words: Vec<&str>) -> Vec<String> {
        let mut stemmer = stemmer::Stemmer::create(self.inner);
        words.into_iter()
            .map(|w| stemmer.stem(w).to_string())
            .collect()
    }
}
```

### 5.3 Python Shim Template (tokenize.py)

```python
"""
fastnltk.tokenize — drop-in replacement for nltk.tokenize.
Delegates to compiled Rust extension where available,
falls back to original nltk.tokenize for unimplemented pieces.
"""

from fastnltk._rust import (
    sent_tokenize as _rust_sent_tokenize,
    word_tokenize as _rust_word_tokenize,
    PunktSentenceTokenizer as _RustPunktSentenceTokenizer,
    # ... more as implemented
)

# Re-export symbols not yet in Rust
import nltk.tokenize as _nltk_tokenize

# ── Rust-backed functions ─────────────────────────────────

def sent_tokenize(text, language="english"):
    """Sentence tokenization (Rust-accelerated)."""
    return _rust_sent_tokenize(text, language)


def word_tokenize(text, language="english", preserve_line=False):
    """Word tokenization (Rust-accelerated)."""
    return _rust_word_tokenize(text, language, preserve_line)


# ── Rust-backed classes ──────────────────────────────────

class PunktSentenceTokenizer:
    """Rust-accelerated Punkt sentence tokenizer."""
    def __init__(self, train_text=None, language="english"):
        self._impl = _RustPunktSentenceTokenizer(language)
        if train_text is not None:
            self._impl.train(train_text)
    def tokenize(self, text):
        return list(self._impl.tokenize(text))
    def span_tokenize(self, text):
        return list(self._impl.span_tokenize(text))
    # ... proxy all methods


# ── Fallback to NLTK ─────────────────────────────────────

TextTilingTokenizer = _nltk_tokenize.TextTilingTokenizer
TweetTokenizer = _nltk_tokenize.TweetTokenizer
# ... every unported symbol

__all__ = [
    'sent_tokenize', 'word_tokenize',
    'PunktSentenceTokenizer', 'PunktTokenizer',
    'RegexpTokenizer', 'WhitespaceTokenizer', ...
]
```

### 5.4 Module Registration (lib.rs)

```rust
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Tokenize
    m.add_function(wrap_pyfunction!(tokenize::sent_tokenize_py, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize::word_tokenize_py, m)?)?;
    m.add_class::<tokenize::PunktSentenceTokenizer>()?;
    // ... etc

    // Stem
    m.add_class::<stem::SnowballStemmer>()?;
    m.add_class::<stem::PorterStemmer>()?;
    // ... etc

    // Tag
    m.add_class::<tag::PerceptronTagger>()?;
    m.add_class::<tag::TnT>()?;
    // ... etc

    // Classify
    m.add_class::<classify::NaiveBayesClassifier>()?;
    // ... etc

    // Probability
    m.add_class::<probability::FreqDist>()?;
    // ... etc

    Ok(())
}
```

---

## 6. Best Practices from Professional Rust+Python Projects

### 6.1 Project Structure

| Practice | Source | Why |
|---|---|---|
| **`crate-type = ["cdylib", "rlib"]`** | pydantic-core | rlib allows `cargo test` and Rust benchmarks without Python. cdylib produces the .pyd/.so. |
| **`python-source` config** | pydantic-core, tokenizers | maturin puts Python source in a subdir, not at repo root. Keeps repo organized. |
| **Module naming: `fastnltk._rust`** | pydantic-core (`pydantic_core._pydantic_core`) | Underscore prefix makes it clear this is the internal compiled module. Python shim re-exports cleanly. |
| **Type stubs (`.pyi` files)** | pydantic-core | IDE autocomplete + type checking for Rust-exported functions. Verify with `mypy.stubtest`. |
| **Separate Rust tests (`#[cfg(test)]`)** | universal (Polars, rustling, vtext) | Rust unit tests alongside code. Run with `cargo test`. |
| **`rust-toolchain.toml`** | pydantic-core, cryptography | Pins MSRV. Prevents accidental breakage from Rust nightly. |

### 6.2 Build & CI

| Practice | Source | Why |
|---|---|---|
| **Use `uv`** for Python management | pydantic-core | 10-100x faster than pip. Lockfile support. `uv sync --group all`. |
| **`make` targets** (`build-dev`, `build-prod`, `lint`, `test`, `format`, `clean`) | pydantic-core, Polars | Standardized dev workflow. One `make all` = build + lint + test. |
| **`maturin develop --uv`** | pydantic-core | Fastest dev iteration: rebuilds Rust, updates venv in place. |
| **`LTO = "fat"`, `codegen-units = 1`, `strip = true`** in release | pydantic-core | Maximizes binary optimization. Critical for NLP hot paths. |
| **`abi3-py38` feature** | pydantic-core, tokenizers | One wheel covers CPython 3.8-3.13. No need to build per-version. |
| **CI test matrix**: 3.8-3.13, PyPy, GraalPy | pydantic-core, Polars | High confidence across interpreters. PyPy catches edge cases. |
| **Platform matrix**: ubuntu + macos + windows | universal | Must test Windows for NLTK compat (path handling, encoding). |
| **`continue-on-error` for free-threaded Python** | pydantic-core | Free-threaded Python 3.13t is experimental. Don't block CI on it. |
| **Rust coverage via `cargo-llvm-cov`** | pydantic-core | Rust code coverage separate from Python. `cargo llvm-cov --codecov`. |
| **Benchmark tracking** | pydantic-core (Codspeed) | Prevents regressions. Run on PRs. |

### 6.3 Rust Code Style

| Practice | Source | Why |
|---|---|---|
| **`OnceLock` / `OnceCell` for lazy model loading** | pydantic-core, rustling | Thread-safe one-time init of tokenizers/taggers. `OnceLock<RwLock<HashMap<...>>>`. |
| **`parking_lot::RwLock` over `std::sync::RwLock`** | rustling, popular consensus | 3-5x faster on contended reads (the common case for model lookup). |
| **`hashbrown::HashMap` over `std::collections::HashMap`** | rustling, vtext | 10-15% faster in hot loops. Drop-in replacement. |
| **`rustc-hash::FxHashMap` for small string-keyed maps** | rustling | FNV hasher faster than SipHash for NLP feature maps. |
| **Release GIL during CPU-bound work** | universal | `Python::allow_threads(|| { ... })`. Other Python threads not blocked. |
| **Python regex flags → Rust RegexBuilder flags** | nltk_rs (plutonium-guy) | Pattern for mapping `re.UNICODE | re.MULTILINE` to Rust equivalents. |
| **`enum_dispatch` for polymorphic dispatch** (if needed) | pydantic-core | Zero-cost trait dispatch for validator/tokenizer variants. |
| **Error mapping: Rust error → `PyErr`** | universal | `fn to_py_error(e: MyError) -> PyErr` converts domain errors. |
| **`#[pyo3(signature = (...))]`** | universal | Explicit default args. Avoids Python `inspect` overhead. |

### 6.4 Python Package Quality

| Practice | Source | Why |
|---|---|---|
| **`ruff` for linting + formatting** | pydantic-core, tokenizers | Fast, comprehensive. Replace flake8 + isort + black. |
| **`pre-commit` hooks** | pydantic-core | Enforce `ruff`, `cargo fmt`, `cargo clippy` before every commit. |
| **`mypy` stubtest** | pydantic-core | Verify `.pyi` stubs match real Rust exports. Prevents drift. |
| **`hypothesis` property-based tests** | pydantic-core | Generate random inputs, ensure output matches NLTK. Catches edge cases. |
| **Max line length 100** | consensus | Rustfmt default + ruff 100. Readable in side-by-side diffs. |
| **Single quotes in Python** | pydantic-core | Matches `ruff format` default. |
| **`pytest-timeout`** | pydantic-core | Prevents infinite loops in CI. 60s default. |

### 6.5 Release Engineering

| Practice | Source | Why |
|---|---|---|
| **`maturin build --release --out dist`** | universal | Builds wheels. Upload with `maturin publish`. |
| **ManyLinux 2_17 wheels** | maturin default | Broad Linux compatibility. |
| **`abi3-py38` wheels** | tokenizers, cryptography | Single wheel for all 3.8+. Smaller release matrix. |
| **PyPI trusted publishing (OIDC)** | modern standard | No API key. GitHub → PyPI trust relationship. |
| **GitHub Actions release workflow** | pydantic-core, tokenizers, cryptography | Tag → build → publish automated. |
| **Release drafter** | Polars | Auto-generates changelog from PR labels. |

### 6.6 Model & Data Management

| Practice | Source | Why |
|---|---|---|
| **FlatBuffers for model serialization** | rustling | Zero-deserialization access. Faster than JSON/protobuf. |
| **gzip-compressed JSON** | rustling (fallback) | Simpler than FlatBuffers for dev. Good enough for small models. |
| **`serde` + `bincode` for model cache** | consensus | Fast binary format. Compatible with `serde` derive. |
| **Lazy loading with `OnceLock`** | rustling, pydantic-core | Don't load models until first use. Split startup time. |
| **NLTK pickle → Rust struct converter** | custom | Ship `scripts/convert_models.py`. Run once per model. |

---

## 7. Testing Strategy

### 7.1 Test Hierarchy

| Layer | Tool | What |
|---|---|---|
| Unit tests (Rust) | `cargo test` | Internal Rust logic: tokenization rules, stemmer algorithms, Viterbi, greedy tagger |
| Integration tests (Python) | `pytest` | Same inputs as NLTK, same outputs expected |
| Compatibility suite | `pytest` + NLTK's test dir | Run NLTK's own test suite against fastNLTK (`sys.modules['nltk'] = fastnltk` monkeypatch) |
| Property-based | `hypothesis` | Random texts, ensure output shape matches NLTK |
| Benchmarks | `pytest-benchmark` | Track speedups vs NLTK. CI alert on regression. |

### 7.2 Running NLTK's Tests Against fastNLTK

```python
# conftest.py — monkeypatch nltk → fastnltk
import sys
import nltk

import fastnltk
sys.modules['nltk'] = fastnltk
sys.modules['nltk.tokenize'] = fastnltk.tokenize
sys.modules['nltk.tag'] = fastnltk.tag
# ... etc
```

Then: `pytest nltk_test_suite/test/unit/test_tokenize.py`

### 7.3 Expected Benchmarks

| Operation | NLTK (ms) | fastNLTK (ms) | Speedup |
|---|---|---|---|
| `word_tokenize("small")` | ~1.5 | ~0.1 | 15x |
| `word_tokenize("medium")` | ~250 | ~12 | 20x |
| `word_tokenize("large")` | ~25,000 | ~800 | 31x |
| `sent_tokenize("medium")` | ~300 | ~10 | 30x |
| `SnowballStemmer().stem(text)` | ~500 | ~25 | 20x |
| `pos_tag(sentences)` | ~150 | ~25 | 6x |
| `NaiveBayesClassifier.train()` | ~1,000 | ~200 | 5x |
| `BigramCollocationFinder.from_words()` | ~200 | ~15 | 13x |
| LM generation (1000 tokens) | ~500 | ~15 | 33x |
| **Full pipeline** (tokenize→tag→chunk→NE) | ~2,000 | ~250 | 8x |

---

## 8. CI/CD Pipeline

### 8.1 GitHub Actions CI

```yaml
name: ci
on: [push, pull_request]

jobs:
  test:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.8", "3.9", "3.10", "3.11", "3.12", "3.13", "pypy3.10"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - uses: PyO3/maturin-action@v1
        with:
          manylinux: auto
          args: --release --out dist
      - run: pip install dist/fastnltk-*.whl
      - run: pip install nltk pytest hypothesis
      - run: python -m nltk.downloader punkt averaged_perceptron_tagger
      - run: pytest tests/ -v
      - run: pytest nltk_test_suite/test/unit/test_tokenize.py -v  # compat check

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy, rustfmt
      - run: cargo fmt --all -- --check
      - run: cargo clippy --tests -- -D warnings
      - uses: astral-sh/ruff-action@v1
        with:
          args: check fastnltk/ tests/

  test-rust:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test

  benchmark:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: PyO3/maturin-action@v1
        with:
          manylinux: auto
          args: --release --out dist
      - run: pip install dist/fastnltk-*.whl
      - run: pip install nltk pytest-benchmark
      - run: python -m nltk.downloader punkt averaged_perceptron_tagger
      - run: pytest benchmarks/ --benchmark-json output.json
      - uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: pytest
          output-file-path: output.json
          alert-threshold: '200%'
```

### 8.2 Release Workflow

```yaml
name: release
on:
  push:
    tags: ["v*"]

jobs:
  build-wheels:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.8", "3.9", "3.10", "3.11", "3.12", "3.13"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: PyO3/maturin-action@v1
        with:
          manylinux: auto
          args: --release --out dist
      - uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}-${{ matrix.python-version }}
          path: dist/*.whl

  publish:
    needs: build-wheels
    runs-on: ubuntu-latest
    permissions:
      id-token: write
    steps:
      - uses: actions/download-artifact@v4
      - uses: PyO3/maturin-action@v1
        env:
          MATURIN_PYPI_TOKEN: ${{ secrets.PYPI_TOKEN }}
        with:
          command: publish
          args: --skip-existing
```

---

## 9. License Compatibility

All selected dependencies are MIT or Apache-2.0 — compatible with our Apache-2.0 release:

| Crate | License | Compatible |
|---|---|---|
| `pyo3` | Apache-2.0 | ✅ |
| `regex` | MIT/Apache-2.0 | ✅ |
| `unicode-segmentation` | MIT/Apache-2.0 | ✅ |
| `rust-stemmers` | MIT | ✅ |
| `rustling` | MIT | ✅ |
| `hashbrown` | MIT/Apache-2.0 | ✅ |
| `rustc-hash` | Apache-2.0/MIT | ✅ |
| `parking_lot` | Apache-2.0/MIT | ✅ |
| `serde` + `bincode` | MIT/Apache-2.0 | ✅ |
| `rayon` | Apache-2.0/MIT | ✅ |
| `whatlang` (optional) | MIT | ✅ |
| vtext (ported code) | Apache-2.0 | ✅ (same license) |
| nltk_rs (study only) | LGPL-3.0 | ❌ — cannot copy code |

---

## 10. Release Roadmap

| Version | Scope | Status |
|---|---|---|
| **v0.1.0** | Core scaffold + **tokenization** (all tokenizers). `word_tokenize`/`sent_tokenize` 10-50x faster. | ✅ Complete |
| **v0.2.0** | **Stemming** (Snowball, Porter, Lancaster, all). **String metrics** (edit_distance, jaro, etc.). | ✅ Complete |
| **v0.3.0** | **POS tagging** (Perceptron, TnT). `pos_tag` 5-6x faster. | ✅ Complete |
| **v0.4.0** | **Classification** (NaiveBayes, MaxEnt, TextCat, PositiveNB). **Collocations**. **FreqDist**. | ✅ Complete |
| **v0.5.0** | **Language models** (MLE, Lidstone, Laplace). **VADER sentiment**. **BLEU scores**. | ✅ Complete |
| **v0.6.0** | **Chunking** (RegexpParser). WordNetLemmatizer. Full API parity with Python shims. | ✅ Complete |
| **v1.0.0** | All NLTK tests pass. Sequential taggers, ARLSTem, HMM tagger. PyPI release. | 📋 Planned |

---

## 11. Appendix: NLTK Module Coverage Matrix

| NLTK Module | fastNLTK Strategy | Est Speedup | Priority | Status | Est Rust LoC |
|---|---|---|---|---|---|
| `nltk.tokenize` | **Rust rewrite** (dep: regex, unicode-segmentation) | 10-50x | **P0** | ✅ Complete | ~3,500 |
| `nltk.stem` | **Rust rewrite** (dep: rust-stemmers) | 10-20x | **P0** | ✅ Complete | ~1,000 |
| `nltk.tag` | **Rust** Perceptron + TnT; sequential/HMM pending | 3-10x | **P0** | ✅ Partial | ~750 |
| `nltk.classify` | **Rust rewrite** (NB, MaxEnt, TextCat, PositiveNB) | 3-50x | **P1** | ✅ Complete | ~1,000 |
| `nltk.collocations` | **Rust rewrite** | 5-15x | **P1** | ✅ Complete | ~500 |
| `nltk.probability` | **Rust rewrite** (FreqDist, CondFreqDist) | 5-10x | **P1** | ✅ Complete | ~500 |
| `nltk.lm` | **Rust wrapper** around rustling LM (MLE, Lidstone, Laplace); KneserNey/WittenBell shim | 10-39x | **P1** | ✅ Complete | ~400 |
| `nltk.metrics` | Rust rewrite (distance, scores) | 5-50x | **P1** | ✅ Complete | ~700 |
| `nltk.sentiment` | Rust: VADER | 3-5x | **P2** | ✅ Complete | ~200 |
| `nltk.translate` | Rust: BLEU/corpus BLEU | 5-10x | **P2** | ✅ Complete | ~150 |
| `nltk.chunk` | Rust: RegexpParser; NE shim | 3-5x | **P2** | ✅ Complete | ~300 |
| `nltk.parse` | Shim (pure Python) | 1x | **P3** | ✅ Shim | 0 |
| `nltk.tree` | Shim (pure Python) | 1x | **P3** | ✅ Shim | 0 |
| `nltk.corpus` | Shim → nltk.corpus | 1x | **P3** | ✅ Shim | 0 |
| `nltk.sem` | Shim | 1x | **P3** | ✅ Shim | 0 |
| `nltk.inference` | Shim | 1x | **P3** | ✅ Shim | 0 |
| `nltk.cluster` | Shim | 1x | **P3** | ✅ Shim | 0 |
| `nltk.ccg` | Shim | 1x | **P4** | ✅ Shim | 0 |
| `nltk.chat` | Shim | 1x | **P4** | ✅ Shim | 0 |
| `nltk.twitter` | Shim | 1x | **P4** | ✅ Shim | 0 |
| `nltk.draw` / `nltk.app` | **Skip** (GUI) | — | **Skip** | — | 0 |
| `nltk.downloader` | Wrap nltk.downloader | 1x | **P3** | ✅ Shim | 0 |
| `nltk.data` | **Rust data loader** + wrap nltk.data | 1x | **P0** | ✅ Complete | ~300 |

**Total Rust LoC** (core NLP): ~8,500
**Total Python LoC** (shims + wrappers): ~5,000
**LoC saved by crates**: ~8,800 (58% reduction vs. rewriting everything from scratch)

All numbers replace **154K Python LoC** of NLTK with **~13K LoC** of Rust + Python shim, delivering **5-50x speedup** on hot paths.

---

## 12. Development Process: One Function at a Time

### 12.1 Phases of Each Function

Every NLTK function/class we port follows this exact workflow:

```
Boilerplate → Pick Function → Port → Test → Benchmark → Record → Merge → Next
```

### 12.2 Phase A: Create Boilerplate First

Before any porting, build the full project scaffold (Section 1). Verify:

```bash
# 1. maturin builds and _rust module is importable
maturin develop --uv --release
python -c "from fastnltk._rust import sent_tokenize; print('rust extension loaded')"

# 2. make targets work
make build-dev
make lint-python   # must pass clean
make lint-rust     # must pass clean

# 3. CI works (push a no-op commit)
git commit --allow-empty -m "chore: scaffold" && git push
```

**Gate**: Boilerplate is done when `pip install -e .` + `from fastnltk import word_tokenize` works (even if it falls back to NLTK).

### 12.3 Phase B: Pick One Function

Select the next function from the priority list (Sections 4, 11). Rules:

- **One function at a time.** No parallel branches. Never start a second function until the first is merged.
- **Start with the simplest** in each module. For tokenization: `SpaceTokenizer` → `RegexpTokenizer` → `WordPunctTokenizer` → `TreebankWordTokenizer` → `PunktSentenceTokenizer`.
- **Each function gets its own branch**: `git checkout -b feat/sent-tokenize`

### 12.4 Phase C: Port to Rust

1. Add Rust implementation in `src/tokenize/<function>.rs`
2. Register in `src/lib.rs` via `#[pymodule]`
3. Add PyO3 `#[pyfunction]` or `#[pyclass]` wrapper
4. Ensure GIL is released during computation

**Minimal first pass**: Get the common case right. Edge cases are handled in Phase D via test-driven fixes.

### 12.5 Phase D: Test Against NLTK

Every function must pass:

**Rust unit tests** (`#[cfg(test)]` in the same .rs file):
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sent_tokenize_basic() {
        let result = sent_tokenize("Hello world. This is a test.");
        assert_eq!(result, vec!["Hello world. ", "This is a test."]);
    }

    #[test]
    fn test_sent_tokenize_abbreviations() {
        let result = sent_tokenize("Mr. Smith went home. He ate dinner.");
        assert_eq!(result, vec!["Mr. Smith went home. ", "He ate dinner."]);
    }
}
```
Run: `cargo test` — must pass.

**Python integration tests** (`tests/test_tokenize.py`):
```python
import pytest
from fastnltk.tokenize import sent_tokenize
import nltk

# Direct comparison: same input, same output
TEXT = "Mr. Smith went to Washington. He met Dr. Jones. They discussed AI."

def test_sent_tokenize_matches_nltk():
    expected = nltk.tokenize.sent_tokenize(TEXT)
    result = sent_tokenize(TEXT)
    assert result == expected, f"Mismatch!\n  NLTK: {expected}\n  fastNLTK: {result}"

# Edge cases
def test_sent_tokenize_empty():
    assert sent_tokenize("") == []

def test_sent_tokenize_no_punctuation():
    assert sent_tokenize("Hello world") == ["Hello world"]

# Property-based test (hypothesis)
from hypothesis import given, strategies as st

@given(st.text(max_size=500))
def test_sent_tokenize_same_as_nltk(text):
    # Skip inputs that crash NLTK
    try:
        expected = nltk.tokenize.sent_tokenize(text)
        result = sent_tokenize(text)
        assert result == expected
    except Exception as e:
        pytest.skip(f"NLTK raised: {e}")
```
Run: `pytest tests/test_tokenize.py -v` — must pass all tests.

**NLTK compat test** (if applicable):
```bash
# Run NLTK's own test suite against our implementation
pytest nltk_test_suite/test/unit/test_tokenize.py -v
```

**Gate**: All tests pass on ubuntu + macos + windows locally or via CI draft PR.

### 12.6 Phase E: Benchmark & Record

Every function gets a benchmark file in `benchmarks/`:

```python
# benchmarks/tokenize_bench.py
import pytest
import nltk
import fastnltk

# Test data (realistic corpus excerpts)
TEXTS = {
    "tiny": "Hello world. This is a test.",                        # ~30 chars
    "small": open("benchmarks/data/paragraph.txt").read(),        # ~1KB
    "medium": open("benchmarks/data/article.txt").read(),         # ~50KB
    "large": open("benchmarks/data/moby_dick.txt").read(),        # ~1.2MB
}

@pytest.mark.parametrize("size", TEXTS.keys())
@pytest.mark.benchmark(group="sent_tokenize")
def test_sent_tokenize_nltk(benchmark, size):
    text = TEXTS[size]
    result = benchmark(nltk.tokenize.sent_tokenize, text)

@pytest.mark.parametrize("size", TEXTS.keys())
@pytest.mark.benchmark(group="sent_tokenize")
def test_sent_tokenize_fastnltk(benchmark, size):
    text = TEXTS[size]
    result = benchmark(fastnltk.tokenize.sent_tokenize, text)
```

Run benchmarks and capture results:
```bash
pytest benchmarks/tokenize_bench.py --benchmark-json benchmarks/results/sent_tokenize.json
```

**Read benchmark results into README table**:

Use a script to compute speedup and update the table:
```python
# scripts/update_benchmark_table.py
"""Parse --benchmark-json output and update README benchmark table."""
import json
import re
from pathlib import Path

RESULTS_DIR = Path("benchmarks/results")

def extract_speedups(benchmark_file):
    with open(RESULTS_DIR / benchmark_file) as f:
        data = json.load(f)
    
    benchmarks = data["benchmarks"]
    # Group by param (text size)
    by_param = {}
    for b in benchmarks:
        param = b["params"]["size"]
        group = b["group"]
        if param not in by_param:
            by_param[param] = {}
        if "nltk" in b["name"]:
            by_param[param]["nltk_ms"] = b["stats"]["mean"] * 1000
        else:
            by_param[param]["fastnltk_ms"] = b["stats"]["mean"] * 1000
    
    results = {}
    for param, vals in by_param.items():
        nltk_ms = vals.get("nltk_ms", 0)
        fastnltk_ms = vals.get("fastnltk_ms", 0)
        speedup = round(nltk_ms / fastnltk_ms, 1) if fastnltk_ms > 0 else "N/A"
        results[param] = {
            "nltk_ms": round(nltk_ms, 2),
            "fastnltk_ms": round(fastnltk_ms, 2),
            "speedup": speedup,
        }
    return results
```

### 12.7 Phase F: Update Benchmark Table in README

The README has a **single source of truth** table for all benchmarks:

```markdown
## Performance Benchmarks

Measured on [hardware description, e.g. Intel i7-12700, 32GB RAM, Ubuntu 24.04].
All benchmarks compare `fastnltk` against `nltk` on identical inputs.
Times are mean of 10+ runs. Last updated: 2026-07-13.

### Tokenization

| Function | Input Size | NLTK (ms) | fastNLTK (ms) | Speedup | Date Added |
|---|---|---|---|---|---|
| `sent_tokenize` | tiny (30B) | 0.12 | 0.01 | 12.0x | 2026-07-13 |
| `sent_tokenize` | small (1KB) | 1.50 | 0.08 | 18.8x | 2026-07-13 |
| `sent_tokenize` | medium (50KB) | 58.20 | 2.10 | 27.7x | 2026-07-13 |
| `sent_tokenize` | large (1.2MB) | 1,420.00 | 45.10 | 31.5x | 2026-07-13 |
| `word_tokenize` | tiny (30B) | 0.15 | 0.01 | 15.0x | 2026-07-14 |
| `word_tokenize` | medium (50KB) | 72.10 | 3.80 | 19.0x | 2026-07-14 |
| `SpaceTokenizer.tokenize` | medium (50KB) | 8.40 | 0.20 | 42.0x | 2026-07-15 |

### Stemming

| Function | Input Size | NLTK (ms) | fastNLTK (ms) | Speedup | Date Added |
|---|---|---|---|---|---|
| `SnowballStemmer.stem` | 10K words | 45.20 | 2.30 | 19.7x | 2026-07-20 |
| `PorterStemmer.stem` | 10K words | 38.10 | 2.80 | 13.6x | 2026-07-21 |

### POS Tagging

| Function | Input Size | NLTK (ms) | fastNLTK (ms) | Speedup | Date Added |
|---|---|---|---|---|---|
| `pos_tag` | 100 sentences | 25.40 | 4.50 | 5.6x | 2026-08-01 |
| `pos_tag` | 1000 sentences | 248.10 | 39.80 | 6.2x | 2026-08-01 |
| `PerceptronTagger.tag` | 100 sentences | 18.90 | 3.10 | 6.1x | 2026-08-01 |

### Full Pipeline

| Pipeline | NLTK (ms) | fastNLTK (ms) | Speedup | Date Added |
|---|---|---|---|---|
| tokenize → tag → chunk → NE | 1,850.00 | 210.00 | 8.8x | 2026-08-15 |
```

**Rules for the table**:

1. **Every merged function adds a row** or multiple rows (one per input size).
2. **Date column** — track when each benchmark was first recorded.
3. **If performance degrades** (e.g., after refactoring), update the row. The date shows most recent measurement.
4. **Hardware note at top** — so readers can contextualize absolute numbers.
5. **Input sizes are standardized**: tiny (~30B), small (~1KB), medium (~50KB), large (~1.2MB).

### 12.8 Phase G: Merge & Announce

```bash
# Final checklist
cargo test --all                          # All Rust tests pass
pytest tests/ -v                          # All Python tests pass
pytest benchmarks/ --benchmark-disable    # Benchmarks don't crash (skip timing)
make lint                                 # Zero warnings

# Update benchmark table
python scripts/update_benchmark_table.py benchmarks/results/sent_tokenize.json

# Commit
git add -A
git commit -m "feat: add sent_tokenize (31.5x speedup on large inputs)"
git push origin feat/sent-tokenize
# → Open PR, CI runs, merge to main
```

**Post-merge**:
- Tag benchmark data: `benchmarks/results/sent_tokenize_v1.json` is archived
- The README table reflects the new merged function
- CI benchmark job runs on `main` weekly and alerts if speedup drops >20%

### 12.9 Development Rhythm Summary

```
┌─────────────────────────────────────────────────────────┐
│            One Function Development Cycle               │
├─────────────────────────────────────────────────────────┤
│ 1. git checkout -b feat/<function-name>                 │
│ 2. Implement Rust core in src/<module>/<fn>.rs          │
│ 3. Register PyO3 wrapper in lib.rs                      │
│ 4. Write Rust unit tests (#[cfg(test)])                 │
│ 5. cargo test (loop until green)                        │
│ 6. Write Python integration tests (tests/test_*.py)     │
│ 7. pytest tests/ (loop until green)                     │
│ 8. Write benchmark (benchmarks/<module>_bench.py)       │
│ 9. pytest benchmarks/ --benchmark-json output.json      │
│ 10. Extract speedup, update README table                │
│ 11. make lint (zero warnings)                           │
│ 12. git commit + push → PR → merge                      │
│ 13. Archive benchmark JSON: cp output.json archive/     │
│ 14. Pick next function from priority list               │
│                                                         │
│ Typical cycle: 1-3 days per simple function              │
│                 3-7 days for complex (Punkt, MaxEnt)     │
└─────────────────────────────────────────────────────────┘
```

### 12.10 Benchmark Data Archive

All raw benchmark JSON files are committed to the repo:

```
benchmarks/
├── data/                          # Standard test corpus files
│   ├── paragraph.txt              # ~1KB
│   ├── article.txt                # ~50KB
│   ├── moby_dick.txt              # ~1.2MB
│   └── ...
├── results/                       # Human-readable JSON
│   ├── sent_tokenize_v1.json
│   ├── sent_tokenize_v2.json      # Updated after optimization
│   ├── word_tokenize_v1.json
│   ├── snowball_stemmer_v1.json
│   ├── pos_tag_v1.json
│   └── ...
├── archive/                       # Historical data (kept forever)
│   ├── 2026-07-13_sent_tokenize.json
│   ├── 2026-07-14_word_tokenize.json
│   └── ...
├── tokenize_bench.py
├── stem_bench.py
├── tag_bench.py
├── collocations_bench.py
└── full_pipeline_bench.py
```

**Why track benchmarks in the README?**

1. **Transparency** — users see exactly what speedup to expect before installing.
2. **Progress tracking** — visualize the project filling out as each row appears.
3. **Regression detection** — if a refactor reduces speedup, the row changes. `git diff` catches it.
4. **Motivation** — watching `sent_tokenize` go from 12x to 31x with each optimization is satisfying.

### 12.11 First Function Walkthrough: `SpaceTokenizer`

As a concrete example, here's the full cycle for the simplest tokenizer:

```rust
// src/tokenize/simple.rs

#[pyclass(name = "SpaceTokenizer", module = "fastnltk._rust")]
pub struct SpaceTokenizer;

#[pymethods]
impl SpaceTokenizer {
    #[new]
    fn new() -> Self {
        Self
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split(' ')
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_space_tokenizer_basic() {
        let tok = SpaceTokenizer::new();
        assert_eq!(tok.tokenize("a b c"), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_space_tokenizer_empty() {
        let tok = SpaceTokenizer::new();
        assert!(tok.tokenize("").is_empty());
    }

    #[test]
    fn test_space_tokenizer_multiple_spaces() {
        let tok = SpaceTokenizer::new();
        assert_eq!(tok.tokenize("a  b"), vec!["a", "b"]);
    }
}
```

```python
# tests/test_simple_tokenizers.py
import pytest
from fastnltk.tokenize.simple import SpaceTokenizer
import nltk

def test_space_tokenizer_matches_nltk():
    text = "a  b   c d"
    expected = nltk.tokenize.SpaceTokenizer().tokenize(text)
    result = SpaceTokenizer().tokenize(text)
    assert result == expected


def test_space_tokenizer_empty():
    assert SpaceTokenizer().tokenize("") == []


def test_space_tokenizer_no_spaces():
    assert SpaceTokenizer().tokenize("abc") == ["abc"]
```

Cycle time for `SpaceTokenizer`: **~30 minutes** from `git checkout -b` to merge PR. This is the quick win that proves the pipeline works, before tackling harder functions like Punkt.
