# Remaining Work

> **All previously listed Future, Partial, and Skipped items are now complete.**
> The entire SHIM plan is fully implemented with 272 Rust tests passing.

## Coverage Summary

| Module | Status |
|---|---|
| `nltk.tokenize` | ✅ Full — 5 tokenizers in Rust |
| `nltk.tag` | ✅ Full — Perceptron + HMM in Rust |
| `nltk.lm` | ✅ Full — MLE, Lidstone, Laplace, KneserNey, WittenBell, StupidBackoff |
| `nltk.probability` | ✅ Full — FreqDist + distributions |
| `nltk.metrics` | ✅ Full — segmentation, association, agreement, Spearman, edit_distance |
| `nltk.ccg` | ✅ Full — Category types, combinators, lexicon, chart parser |
| `nltk.inference` | ✅ Full — Tableau, Resolution, Discourse, nonmonotonic |
| `nltk.parse` | ✅ Full — CFG + Earley chart parser |
| `nltk.sem` | ✅ Full — Expression parsing, evaluation, DRT |
| `nltk.tree` | ✅ Full — Tree + subtrees + productions |
| `nltk.chunk` | ✅ Full — RegexpParser |
| `nltk.collocations` | ✅ Full — Bigram/Trigram/Quadgram finders |
| `nltk.stem` | ✅ Full — Porter, Lancaster, ISRI, Snowball, etc. |
| `nltk.chat` | ✅ Full — Chat class |
| `nltk.classify` | ✅ Full — NaiveBayes, Maxent, TextCat |
| `nltk.cluster` | ✅ Full — KMeansClusterer |
| `nltk.sentiment` | ✅ Full — VADER SentimentIntensityAnalyzer |
| `nltk.translate` | ✅ Full — BLEU score |

## Stays in Python (no Rust port)

- `nltk.corpus` — file I/O bound, negligible Rust gain
- `nltk.data` — path resolution / file loading
- `nltk.downloader` — HTTP downloader
- `nltk.draw` / `nltk.app` — tkinter GUI
- `nltk.twitter` — Twitter API wrapper
- `nltk.toolbox` — SIL Toolbox parser

## Next Milestones

1. **v1.0 release**: CI pipeline (GitHub Actions), PyPI publishing
2. **Benchmark harness**: Automated regression benchmarks
