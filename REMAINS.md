# Remaining Work — Future, Partial & Not Yet Ported

> **All previously listed Future and Partial items are now complete.**
> See milestones below for what's next.

---

## ✅ Completed (as of this REMAINS.md update)

### Future items now done

| Item | File | What Was Done |
|---|---|---|
| **CCG lexicon loading** | `src/ccg/lexicon.rs` | `HashMap<String, Vec<Category>>` with file I/O. Registered as PyO3 `CCGLexicon` class. |
| **CCG chart parser** | `src/ccg/chart.rs` | CKY-style CCG chart parser using combinator rules. `CCGChartParser` class. |
| **Discourse thread + DRT bridge** | `src/inference/discourse.rs` | `DiscourseThread` with DRS merge, FOL conversion, yes/no question answering. |
| **Nonmonotonic reasoning** | `src/inference/nonmonotonic.rs` | `DefaultReasoner` (extension computation), `ClosedWorldReasoner` (CWA). |

### Partial items now done

| Item | What Was Changed |
|---|---|
| **Remove `rust_available` fallback checks** | Stripped from all 21 Python shims. Each now directly imports from `_rust`. NLTK fallbacks kept only for runtime data (missing tagger models, punkt data). |

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
| `nltk.twitter` | Twitter API wrapper — network I/O bound | `from nltk.twitter import *` |
| `nltk.classify` | Thin wrappers around sklearn / numpy models | `from nltk.classify import *` |
| `nltk.cluster` | Wrappers for scipy / numpy clustering | `from nltk.cluster import *` |
| `nltk.sentiment` | VADER lexicon + rule-based sent (trivial) | `from nltk.sentiment import *` |
| `nltk.translate` | IBM Model1-3 with EM training — numpy-dependent | `from nltk.translate import *` |
| `nltk.toolbox` | SIL Toolbox data format parser | `from nltk.toolbox import *` |

### Already Ported (Coverage Check)

| Module | Status | Notes |
|---|---|---|
| `nltk.tokenize` | ✅ **Full coverage** | All 5 tokenizers in Rust (SExpr, TokTok, MWE, TextTiling, Punkt) |
| `nltk.tag` | ✅ **Full coverage** | PerceptronTagger + HMMTagger in Rust |
| `nltk.lm` | ✅ **Full coverage** | MLE, Lidstone, Laplace, KneserNey, WittenBell, StupidBackoff |
| `nltk.probability` | ✅ **Full coverage** | FreqDist + all LM-backed distributions |
| `nltk.metrics` | ✅ **Full coverage** | segmentation, association, agreement, Spearman, edit_distance |
| `nltk.ccg` | ✅ **Full coverage** | Category types + combinators + lexicon + chart parser in Rust |
| `nltk.inference` | ✅ **Full coverage** | Tableau + Resolution + Discourse + nonmonotonic in Rust |
| `nltk.parse` | ⚠️ **Partial** | Earley parser in Rust. Other parsers (chart, recursive descent) via NLTK. |
| `nltk.sem` | ✅ **Full coverage** | Expression types + parsing + evaluation + DRT in Rust |
| `nltk.tree` | ✅ **Full coverage** | Tree + ParentedTree + ImmutableTree in Rust |
| `nltk.chunk` | ✅ **Full coverage** | RegexpParser in Rust |
| `nltk.collocations` | ✅ **Full coverage** | BigramCollocationFinder via AssociationMeasures in Rust |
| `nltk.stem` | ✅ **Full coverage** | Porter + Lancaster + ISRI stemmers in Rust |
| `nltk.chat` | ✅ **Full coverage** | Chat class in Rust |

---

## 🔲 Key Future Milestones

| Milestone | Items Needed | Priority |
|---|---|---|
| **v1.0 release** | CI pipeline (GitHub Actions), PyPI publishing, benchmark harness | High |
| **Expand parse coverage** | Chart parser, recursive descent, shift-reduce parsers in Rust | Low |

## 🔲 Skipped Items (Deliberate)

| Item | Reason |
|---|---|
| **NIST tokenizer** | `NISTTokenizer` does not exist in NLTK's `nltk.tokenize` |
| **Prover9/Mace4 wrappers** | External binaries — stay in Python via `from nltk.inference import *` |
