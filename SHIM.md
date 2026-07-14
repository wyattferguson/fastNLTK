# SHIM Plan — Remaining Python → Rust Porting

## Current State

**47 Rust source files** already exist across 18 modules. **21 Python shim files** are complete with full API parity (verified against `nltk-3.9`). This plan documents the remaining candidates for Rust acceleration and the crates available to help.

---

## Crates to Add

| Crate | Version | License | What It Saves | Our Wrapper LoC |
|---|---|---|---|---|
| `nltk-metrics` | 0.4.0 | Apache-2.0 | Segmentation (windowdiff, pk, ghd), association measures (PMI, chi-sq, Dice), agreement (kappa, pi, alpha), Spearman correlation | ~80 |
| `rustling` (already dep) | 0.8 | MIT | HMM (Baum-Welch, Viterbi, Forward) — already in dep but not wired | ~250 |
| `sgmlish` (optional) | 0.2 | MIT/Apache-2.0 | SGML parsing for NIST tokenizer | ~50 |
| `sexpr_parse` (optional) | 2.0.1 | MIT/Apache-2.0 | S-expression parsing for SExpr tokenizer | ~30 |
| `theorem-prover` | 0.1.1 | MIT | First-order logic theorem prover for inference module | ~200 |
| `lazycop` (optional) | git | MIT | Full connection tableau prover for FOL with equality — complex, evaluate standalone | ~500 wrapper |

### Why `nltk-metrics`

A faithful Apache-2.0 Rust port of NLTK's metrics submodules. Published 2026-06, actively maintained. Covers:

```rust
use nltk_metrics::segmentation::{windowdiff, pk, ghd};
use nltk_metrics::association::{BigramAssocMeasures, NgramAssocMeasures};
use nltk_metrics::agreement::{AnnotationTask, kappa, pi, alpha};
use nltk_metrics::spearman::spearman;
```

Saves ~500 LoC of custom Rust. Already verified against NLTK (Python is the oracle).

### Why `rustling` (already depended)

We already have `rustling = "0.8"` in Cargo.toml but only use `rustling::lm::{MLE, Lidstone, Laplace}`. The crate also provides:

- `rustling::hmm::HiddenMarkovModel` — Baum-Welch EM, Viterbi decoding, Forward scoring, supervised/unsupervised training
- `rustling::hmm::BaseHiddenMarkovModel` trait — full HMM interface

No new dependency needed. Just add `src/tag/hmm.rs` wrapping what's already in the dep graph.

---

## Rust Module — HMM Tagger

### `src/tag/hmm.rs` — ~250 LoC

Wrap `rustling::hmm::HiddenMarkovModel` into NLTK-compatible class.

```rust
#[pyclass(name = "HiddenMarkovModelTagger", module = "fastnltk._rust")]
pub struct HiddenMarkovModelTagger {
    inner: RustHiddenMarkovModel<String>,
    trained: bool,
}
```

Register in `src/tag/mod.rs`:
```rust
pub mod hmm;
m.add_class::<hmm::HiddenMarkovModelTagger>()?;
```

---

## Rust Module — KneserNey/WittenBell/StupidBackoff

### Extend `src/lm.rs` — ~300 LoC

rustling provides MLE/Lidstone/Laplace but NOT KneserNey, WittenBell, or StupidBackoff. Write from scratch using `hashbrown::HashMap` for ngram counts:

| Class | Algorithm | Est LoC |
|---|---|---|
| `KneserNeyInterpolated` | Modified Kneser-Ney smoothing with continuation counts | ~150 |
| `WittenBellInterpolated` | Witten-Bell discounting | ~100 |
| `StupidBackoff` | Simple backoff with alpha scaling factor | ~50 |

These already exist as Python shim classes that fall back to NLTK. The Rust equivalents would be 10-39x faster.

---

## Rust Modules — Tokenizers (5 new files)

### 1. `src/tokenize/toktok.rs` — TokTokTokenizer (~200 LoC)

NLTK's TokTok is a regex-based tokenizer that inserts spaces around punctuation.

```
Patterns: word-internal apostrophes, contractions, URLs, punctuation splits
```

Write from scratch using `regex` crate (already depended). Register in `src/tokenize/mod.rs`.

### 2. `src/tokenize/mwe.rs` — MWETokenizer (~250 LoC)

Multi-word expression tokenizer. Given a list of multi-word expressions, merges them into single tokens when found in sequence.

```
Input:  ["New", "York", "is", "big"]
MWE:    ["New York"]
Output: ["New_York", "is", "big"]
```

Uses `Vec` + string matching. No new deps. Register with a `separator` param (default `"_"`).

### 3. `src/tokenize/nist.rs` — NISTTokenizer (~150 LoC)

SGML-like tokenizer for NIST evaluation. Splits on SGML tags, handles `{` `}` `<` `>` specially. Optionally use `sgmlish` crate or write simple state machine with `regex`.

### 4. `src/tokenize/sexpr.rs` — SExprTokenizer (~100 LoC)

S-expression tokenizer. Splits `(foo (bar baz))` → brackets + atoms. Simple state machine over chars; optionally use `sexpr_parse` crate.

### 5. `src/tokenize/texttiling.rs` — TextTilingTokenizer (~350 LoC)

Algorithm port. Steps:
1. Tokenize into pseudo-sentences (fixed-size blocks)
2. Compute lexical similarity between adjacent blocks (cosine similarity of vocab vectors)
3. Find valleys in similarity curve (boundaries)
4. Return segments

No existing Rust crate implements this. Port from NLTK's `nltk.tokenize.texttiling.TextTilingTokenizer`.

---

## Rust Module — CCG (Combinatory Categorial Grammar)

### `src/ccg/` module — ~1,700 LoC in 4 files

NLTK's `ccg` is 1,703 LoC Python across 6 files. Core submodules:

| File | NLTK LoC | What | Est Rust LoC |
|---|---|---|---|
| `chart.rs` | 496 | CCG chart parser — builds on Earley (use `src/parse.rs` infrastructure) | ~400 |
| `combinator.rs` | 340 | Forward/Backward Application, Composition, Type Raising, Coordination | ~250 |
| `lexicon.rs` | 348 | Lexicon loading from CCGbank/simple format + category entries | ~200 |
| `logic.rs` | 151 | Semantic composition via lambda calculus (reuse `src/sem.rs`) | ~100 |
| `api.rs` | 368 | Public API + category types (primitive: N/NP/S/PP; functional: A/B, A\B) | ~200 |
| **Total** | **1,703** | | **~1,150** |

### Implementation strategy

1. **Category types** in `src/ccg/category.rs` — algebraic enum for primitive + functional, matching NLTK's `Category` hierarchy
2. **Combinator rules** in `src/ccg/combinator.rs` — enum dispatch over rule variants, unify + apply
3. **Lexicon** in `src/ccg/lexicon.rs` — `HashMap<String, Vec<Category>>` with file I/O
4. **Chart parser** in `src/ccg/chart.rs` — subclasses our EarleyChartParser with CCG rule application
5. **API** in `src/ccg/mod.rs` — PyO3 classes: `CCGChartParser`, `CCGLexicon`, `Category`

### Crates that help

| Crate | Verdict |
|---|---|
| `montague-core` | **AGPL-3.0** — incompatible. Skip. |
| `ccg` (odashi) | Empty crate (5 LoC). Skip. |
| Our own `parse.rs` + `sem.rs` | Reuse Earley parser + Expression types. Already own. |

**No new deps.** Write from scratch using our existing parser infrastructure.

### Speedup estimate: 3-5x

CCG is mostly symbolic computation (unification, category matching). Rust's algebraic enums + pattern matching give modest but real gains over Python classes + isinstance checks.

---

## Rust Module — Inference (Tableau + Resolution)

### `src/inference/` module — ~2,200 LoC in 4 files

NLTK's `inference` is 4,253 LoC Python. ~1,800 LoC stay in Python (Prover9/Mace wrappers). ~2,500 LoC are algorithmic and Rust-viable:

| File | NLTK LoC | What | Port to Rust? | Est Rust LoC |
|---|---|---|---|---|
| `tableau.rs` | 749 | Connected tableau theorem prover | ✅ Full port | ~500 |
| `resolution.rs` | 788 | Resolution theorem prover (CNF + binary resolution) | ✅ Full port | ~500 |
| `api.rs` | 614 | ProverCommand, Prover base class, ProverResult | ✅ Partial port | ~400 |
| `discourse.rs` | 651 | Discourse thread processing for DRT (reuse `src/drt.rs`) | ✅ Partial port | ~400 |
| `nonmonotonic.rs` | 561 | DefaultReasoner, ClosedWorldReasoner | ✅ Partial port | ~400 |
| `prover9.py` | 507 | Wrapper for Prover9/Mace4 external binaries | ❌ Stays Python | — |
| `mace.py` | 383 | Wrapper for Mace4 model finder external binary | ❌ Stays Python | — |
| **Total** | **4,253** | | | **~2,200** |

### Implementation strategy

1. **Tableau prover** in `src/inference/tableau.rs` — connected tableau calculus
   - Clause normalization (CNF for FOL)
   - Connected tableau expansion with unification
   - Backtracking search with regularity + tautology elimination
   - Reuse `src/sem.rs` Expression + Variable types

2. **Resolution prover** in `src/inference/resolution.rs` — binary resolution + factoring
   - Skolemization (reuse `src/sem.rs`)
   - CNF conversion
   - Binary resolution with unification
   - Set-of-support strategy, subsumption deletion

3. **API** in `src/inference/mod.rs` — ProverCommand, ProverResult

4. **Discourse** in `src/inference/discourse.rs` — DRT discourse threads
   - Builds on `src/drt.rs` (already exists)
   - Reading comprehension question answering

5. **Nonmonotonic** in `src/inference/nonmonotonic.rs` — default logic + closed-world
   - DefaultReasoner: extension computation via propositional default rules
   - Custom data structures (extension tracking, justification enumeration)

### Crates that help

| Crate | Verdict |
|---|---|
| `theorem-prover` (0.1.1) | 2,708 LoC Rust, MIT. First-order logic prover with parser. New (Jun 2026), undocumented. **Evaluate as optional dep** — could replace tableau + resolution if API compatible. |
| `lazycop` (git, MIT) | Full connection tableau prover for FOL with equality. Designed as standalone binary with TPTP I/O. **Complex integration** — would need adapter from NLTK's API to TPTP format. Evaluate for future. |
| Our own `sem.rs` + `drt.rs` | Expression types, unification, model evaluation. Already own. |

**Recommended**: Start with from-scratch tableau + resolution using `src/sem.rs` types. Evaluate `theorem-prover` crate as possible replacement in a later iteration.

### Speedup estimate: 4-8x

Tableau and resolution provers do heavy symbolic manipulation (unification, CNF conversion, clause management). Rust's ownership model + Vec-based clause stores avoid Python's allocation overhead on hot paths (backtracking, literal matching).

---

## Rust Modules — Metrics (3 new files)

### 1. `src/metrics/segmentation.rs` — ~50 LoC (wrapper)

Delegate to `nltk_metrics::segmentation`:

```rust
use nltk_metrics::segmentation::{windowdiff, pk, ghd};

#[pyfunction]
fn windowdiff_py(reference: &str, hypothesis: &str, k: usize) -> f64 { ... }
#[pyfunction]
fn pk_py(reference: &str, hypothesis: &str) -> f64 { ... }
```

### 2. `src/metrics/association.rs` — ~80 LoC (wrapper)

Delegate to `nltk_metrics::association`:

```rust
use nltk_metrics::association::BigramAssocMeasures;

#[pyclass]
struct AssociationMeasures {
    ngram_assoc: NgramAssocMeasures,
}
```

### 3. `src/metrics/agreement.rs` — ~100 LoC (wrapper if nltk-metrics used)

Or write from scratch (~300 LoC) if avoiding dep:

```
kappa, pi, alpha (Krippendorff), S, agreement via annotation task
```

---

## Python Shim Updates

No new shim files needed. Update existing shims to import from `_rust` when modules are available:

| Shim | Add Rust import |
|---|---|
| `fastnltk/tokenize.py` | `TokTokTokenizer`, `MWETokenizer`, `NISTTokenizer`, `SExprTokenizer`, `TextTilingTokenizer` |
| `fastnltk/tag.py` | `HiddenMarkovModelTagger` |
| `fastnltk/metrics.py` | `windowdiff`, `pk`, `ghd`, `BigramAssocMeasures`, agreement fns |
| `fastnltk/lm.py` | Already has Rust-backed KneserNey/WittenBell/StupidBackoff — replace fallback with `_rust` import |

---

## Implementation Order

### Phase A — Low-hanging fruit (1-2 days)

| Task | Files | Est LoC | Est Time |
|---|---|---|---|
| SExpr tokenizer | `src/tokenize/sexpr.rs` + mod.rs | 100 | 1 hr |
| NIST tokenizer | `src/tokenize/nist.rs` + mod.rs | 150 | 2 hr |
| TokTok tokenizer | `src/tokenize/toktok.rs` + mod.rs | 200 | 2 hr |
| Add `nltk-metrics` dep | `Cargo.toml` | 1 | 5 min |
| Segmentation metrics (wrapper) | `src/metrics/segmentation.rs` | 50 | 30 min |
| **Subtotal** | | **~500** | **~1 day** |

### Phase B — Medium effort (2-3 days)

| Task | Files | Est LoC | Est Time |
|---|---|---|---|
| MWE tokenizer | `src/tokenize/mwe.rs` + mod.rs | 250 | 3 hr |
| HMM tagger (wrap rustling) | `src/tag/hmm.rs` + mod.rs | 250 | 4 hr |
| Association measures (wrapper) | `src/metrics/association.rs` | 80 | 1 hr |
| Agreement metrics (wrapper/port) | `src/metrics/agreement.rs` | 100 | 2 hr |
| Spearman correlation | `src/metrics/spearman.rs` | 50 | 30 min |
| **Subtotal** | | **~730** | **~2 days** |

### Phase C — Hard effort (3-5 days)

| Task | Files | Est LoC | Est Time |
|---|---|---|---|
| KneserNey in Rust | `src/lm.rs` (extend) | 150 | 1 day |
| WittenBell in Rust | `src/lm.rs` (extend) | 100 | 0.5 day |
| StupidBackoff in Rust | `src/lm.rs` (extend) | 50 | 0.5 day |
| TextTiling tokenizer | `src/tokenize/texttiling.rs` + mod.rs | 350 | 2 days |
| **Subtotal** | | **~650** | **~4 days** |

### Phase D — CCG module (3 days)

| Task | Files | Est LoC | Est Time |
|---|---|---|---|
| Category types + API | `src/ccg/mod.rs`, `src/ccg/api.rs` | 200 | 4 hr |
| Combinator rules | `src/ccg/combinator.rs` | 250 | 6 hr |
| Lexicon loading | `src/ccg/lexicon.rs` | 200 | 4 hr |
| Chart parser | `src/ccg/chart.rs` | 400 | 8 hr |
| Python shim + tests | `fastnltk/ccg.py` + `tests/` | 100 | 4 hr |
| **Subtotal** | | **~1,150** | **~3 days** |

### Phase E — Inference module (5 days)

| Task | Files | Est LoC | Est Time |
|---|---|---|---|
| Tableau prover | `src/inference/tableau.rs` | 500 | 2 days |
| Resolution prover | `src/inference/resolution.rs` | 500 | 2 days |
| API + ProverCommand | `src/inference/mod.rs` | 400 | 4 hr |
| Discourse thread + DRT bridge | `src/inference/discourse.rs` | 400 | 4 hr |
| Nonmonotonic reasoning | `src/inference/nonmonotonic.rs` | 400 | 4 hr |
| Python shim + tests | `fastnltk/inference.py` + `tests/` | 100 | 4 hr |
| **Subtotal** | | **~2,300** | **~5 days** |

### Phase F — Python shim wiring (0.5 day)

| Task | Files | Est LoC | Est Time |
|---|---|---|---|
| Update tokenize.py imports | `fastnltk/tokenize.py` | 15 | 15 min |
| Update tag.py imports | `fastnltk/tag.py` | 5 | 5 min |
| Update metrics.py imports | `fastnltk/metrics.py` | 20 | 20 min |
| Update lm.py imports | `fastnltk/lm.py` | 5 | 5 min |
| Create fastnltk/ccg.py with Rust imports | `fastnltk/ccg.py` | 20 | 15 min |
| Create fastnltk/inference.py with Rust imports | `fastnltk/inference.py` | 20 | 15 min |
| Remove rust_available fallback checks | (every Rust-backed shim) | 50 | 15 min |
| **Subtotal** | | **~135** | **~0.5 day** |

---

## Total Tally

| Category | Files | LoC | Time |
|---|---|---|---|
| New Rust (tokenizers) | 5 rs | ~1,050 | ~3 days |
| New Rust (HMM tagger) | 1 rs | ~250 | ~0.5 day |
| New Rust (CCG) | 5 rs | ~1,150 | ~3 days |
| New Rust (inference) | 4 rs | ~2,200 | ~5 days |
| New Rust (LM extension) | 1 rs (extend) | ~300 | ~2 days |
| New Rust (metrics wrappers) | 3 rs | ~280 | ~0.5 day |
| Python shim updates | 6 py | ~135 | ~0.5 day |
| **Total** | | **~5,365** | **~16 days** |

---

## What Stays in Python (no Rust port)

| Module | Reason | Strategy |
|---|---|---|
| `nltk.twitter` | Twitter API wrapper — readonly, not perf-critical | `from nltk.twitter import *` |
| `nltk.draw` / `nltk.app` | tkinter GUI — by design, not porting | Skip entirely |

---

## Benchmark Targets

| Operation | Current (NLTK fallback) | Target (Rust) | Est Speedup |
|---|---|---|---|
| `TokTokTokenizer.tokenize` (50KB) | ~40 ms | ~1.5 ms | 25x |
| `MWETokenizer.tokenize` (50KB) | ~30 ms | ~3 ms | 10x |
| `TextTilingTokenizer.tokenize` (50KB) | ~500 ms | ~60 ms | 8x |
| `HiddenMarkovModelTagger.tag` (1K words) | ~50 ms | ~8 ms | 6x |
| `KneserNeyInterpolated.score` (10K queries) | ~110 ms | ~18 ms | 6x |
| `CCGChartParser.parse` (20 words) | ~5 ms | ~1 ms | 5x |
| `TableauProver.prove` (medium formula) | ~50 ms | ~8 ms | 6x |
| `ResolutionProver.prove` (medium formula) | ~80 ms | ~15 ms | 5x |
| `windowdiff` (10K chars) | ~2 ms | ~0.05 ms | 40x |
| `pk` (10K chars) | ~2 ms | ~0.05 ms | 40x |

---

## Development Workflow

Per the PLAN.md "One Function at a Time" process:

```bash
git checkout -b feat/toktok-tokenizer
# 1. Implement src/tokenize/toktok.rs
# 2. Register in src/tokenize/mod.rs
# 3. cargo test (Rust unit tests)
# 4. Update fastnltk/tokenize.py (remove fallback)
# 5. pytest tests/ -v (Python integration)
# 6. Update benchmarks/tokenize_bench.py
# 7. make lint
# 8. git commit + push + PR → merge
# 9. Next function
```
