# Remaining Work — Future, Partial & Not Yet Ported

Items extracted from `SHIM.md` + other NLTK modules not yet accelerated.

---

## 🔲 Future / Deferred Items

Items explicitly deferred to a later iteration.

### CCG Module (Phase D)

| Item | File | Est LoC | What's Missing |
|---|---|---|---|
| **Lexicon loading** | `src/ccg/lexicon.rs` | ~200 | `HashMap<String, Vec<Category>>` with file I/O for CCGbank/simple format. NLTK fallback via `from nltk.ccg import *` active. |
| **Chart parser** | `src/ccg/chart.rs` | ~400 | CCG chart parser building on Earley infrastructure. NLTK fallback active. |

### Inference Module (Phase E)

| Item | File | Est LoC | What's Missing |
|---|---|---|---|
| **Discourse thread + DRT bridge** | `src/inference/discourse.rs` | ~400 | Discourse processing for DRT, reading comprehension QA. Builds on `src/drt.rs`. |
| **Nonmonotonic reasoning** | `src/inference/nonmonotonic.rs` | ~400 | `DefaultReasoner`, `ClosedWorldReasoner`. Custom data structures for extension tracking. |

---

## 🔲 Partial / Uncoupled Items

Items where some Rust code exists but is not yet wired to Python.

### Fallback Guards

| Item | Scope | What to Do |
|---|---|---|
| **Remove `rust_available` fallback checks** | All Rust-backed shims (tokenize.py, tag.py, metrics.py, lm.py, ccg.py, inference.py) | Strip `try: from fastnltk._rust import ... / except: rust_available = False` pattern. Currently kept as defensive fallback — safe to remove once Rust extension is guaranteed present in all deployment scenarios. |

---

## 🔲 Skipped Items (Deliberate)

Items considered for Rust port and rejected.

| Item | Reason | Current Status |
|---|---|---|
| **NIST tokenizer** | `NISTTokenizer` does not exist in NLTK's `nltk.tokenize` module | No-op, removed from plan |

---

## ❌ Stays in Python (No Rust Port Planned)

Modules that remain as pure Python re-exports from NLTK.

### I/O & Infrastructure

| Module | Reason for Staying | Strategy |
|---|---|---|
| `nltk.corpus` | File I/O bound, not CPU — Rust gains negligible | `from nltk.corpus import *` |
| `nltk.data` | Path resolution + file loading for NLTK data | `from nltk.data import *` |
| `nltk.downloader` | HTTP downloader with progress bars | `from nltk.downloader import *` |
| `nltk.book` | Textbook convenience wrapper | Skip entirely |

### UI & Visualization

| Module | Reason for Staying | Strategy |
|---|---|---|
| `nltk.draw` | tkinter GUI — by design not portable | Skip entirely |
| `nltk.app` | tkinter GUI applications | Skip entirely |
| `nltk.chart` | Visualization backend for parse trees | Skip entirely |

### External Wrappers

| Module | Reason for Staying | Strategy |
|---|---|---|
| `nltk.twitter` | Twitter API wrapper — network I/O bound, not perf-critical | `from nltk.twitter import *` |
| `nltk.classify` | Thin wrappers around sklearn / numpy models | `from nltk.classify import *` |
| `nltk.cluster` | Wrappers for scipy / numpy clustering | `from nltk.cluster import *` |
| `nltk.sentiment` | VADER lexicon + rule-based sent (trivial) | `from nltk.sentiment import *` |
| `nltk.translate` | IBM Model1-3 with EM training — data-parallel, numpy-dependent | `from nltk.translate import *` |
| `nltk.toolbox` | SIL Toolbox data format parser | `from nltk.toolbox import *` |

### Already Ported (Coverage Check)

| Module | Status | Notes |
|---|---|---|
| `nltk.tokenize` | ✅ **Full coverage** | All 5 tokenizers in Rust (SExpr, TokTok, MWE, TextTiling, Punkt) |
| `nltk.tag` | ✅ **Full coverage** | PerceptronTagger + HMMTagger in Rust |
| `nltk.lm` | ✅ **Full coverage** | MLE, Lidstone, Laplace, KneserNey, WittenBell, StupidBackoff |
| `nltk.probability` | ✅ **Full coverage** | FreqDist + all LM-backed distributions |
| `nltk.metrics` | ✅ **Full coverage** | segmentation, association, agreement, Spearman, edit_distance |
| `nltk.ccg` | ⚠️ **Partial** | Category types + combinator rules in Rust. Lexicon + chart parser fall through to NLTK. |
| `nltk.inference` | ⚠️ **Partial** | Tableau + Resolution provers in Rust. Discourse + nonmonotonic fall through to NLTK. |
| `nltk.parse` | ⚠️ **Partial** | Earley parser in Rust. Other parsers (chart, recursive descent, shift-reduce) fall through. |
| `nltk.sem` | ⚠️ **Partial** | Expression types + valuation in Rust. Inference utilities via NLTK. |
| `nltk.tree` | ✅ **Full coverage** | Tree + ParentedTree + ImmutableTree in Rust |
| `nltk.chunk` | ✅ **Full coverage** | ChunkParser + RegexpChunkParser cascade in Rust |
| `nltk.collocations` | ✅ **Full coverage** | BigramCollocationFinder via AssociationMeasures in Rust |
| `nltk.stem` | ✅ **Full coverage** | Porter + Lancaster + ISRI stemmers in Rust |
| `nltk.chat` | ✅ **Full coverage** | Eliza + other chatbot scripts |

---

## 🔲 Key Future Milestones

| Milestone | Items Needed | Priority |
|---|---|---|
| **v1.0 release** | CI pipeline, PyPI publishing, benchmark harness | High |
| **CCG full support** | `lexicon.rs` + `chart.rs` — eliminates last NLTK fallback in CCG | Medium |
| **Inference full support** | `discourse.rs` + `nonmonotonic.rs` — eliminates last NLTK fallback in inference | Medium |
| **Remove fallback guards** | Strip `rust_available` pattern from all shims | Low (after CI deployment) |
| **NIST tokenizer re-eval** | Verify if newer NLTK versions added it | Low |
