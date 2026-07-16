# Changelog

All notable changes to fastNLTK are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] — 2025-07-16

### Changed
- Clippy cleanup for Rust 1.97: suppressed `cast_precision_loss`/`needless_pass_by_value` at crate level, fixed `or_fun_call`, `branches_sharing_code`, `needless_range_loop`, `let_and_return`, and other style lints
- Benchmark results updated with release build numbers (geo mean 8.3× across 49 benchmarks)
- `.gitignore` now excludes `benchmarks/results/*.json`

## [0.4.0] — 2025-07-15

### Added
- Regression tracking benchmark harness (`benchmarks/` → `python -m benchmarks.run`)
- Multi-language Snowball stemmer support (14 languages)
- ISRI Arabic stemmer (full root extraction)
- ARLSTem / ARLSTem2 Arabic stemmers
- Cistem German stemmer
- RSLP Portuguese stemmer
- BLEU score with N-gram precision (translate module)
- Jaro-Winkler and Dice string similarity (metrics module)
- Grammar-normalized Earley chart parser with parse_sents
- K-means clustering with classify/centroids API
- Eliza-style chatbot pattern matching (chat module)
- CCG category parser
- Collocations: Bigram, Trigram, Quadgram finders
- DRT (Discourse Representation Theory) module
- Inference: Tableau prover, Resolution prover, Discourse
- NLTK-identical test invariances (253+ compatibility tests)
- Benchmark regression CI gate (fails if perf drops >15%)
- bincode-based PerceptronTagger cache (297× faster load)

### Changed
- Moved to own benchmark harness (drop pytest-benchmark for benchmarking)
- Python 3.8+ compatibility via PyO3 abi3-py38
- PerceptronTagger: u64 feature IDs with FxHasher (eliminates String allocation)
- Treebank tokenizer: single-pass char scanner with SIMD memchr3
- Regexp tokenizer: whitespace fast-path via memchr3/SIMD
- Build: sccache, cargo-nextest, parallel codegen in CI
- Removed zstd-sys / bzip2-sys transitive deps (−45s cold build, −10MB)

### Fixed
- Punkt sentence tokenizer lazy-loading from NLTK data
- HMM tagger error propagation to Python exceptions
- WordPunct tokenizer non-ASCII punctuation handling
- Tree bracket parser edge cases (empty labels, unclosed brackets)
- Edit distance transposition flag

## [0.3.0] — 2025-06-20

### Added
- Full NLTK module layout: classify, cluster, metrics, translate, sentiment, collocations
- VADER sentiment analysis in Rust
- Maximum entropy classifier (GIS training)
- Naive Bayes classifier
- Language models: MLE, Laplace, Lidstone, Kneser-Ney, Witten-Bell, StupidBackoff
- ConditionalFreqDist and FreqDist with NLTK-identical API
- WordNet lemmatizer (morphy algorithm)
- Complete tokenizer suite: Punkt, Treebank, Tweet, Regexp, Whitespace, WordPunct, S-Expr, Toktok
- TextTiling tokenizer
- MWE (Multi-Word Expression) tokenizer
- Blankline/Line/Tab/Char tokenizers

### Changed
- Build: maturin 1.7+, PyO3 0.29, Python abi3-py38
- Module layout mirrors NLTK exactly (drop-in import compatibility)
- PerceptronTagger uses tagdict fast-path for common words
- All tokenizers expose span_tokenize API

### Fixed
- Snowball stemmer non-English language handling
- Porter stemmer measure calculation edge cases
- Regexp tokenizer gap-mode correctness

## [0.2.0] — 2025-05-01

### Added
- POS tagging: Averaged Perceptron tagger
- POS tagging: Hidden Markov Model tagger
- POS tagging: DefaultTagger, UnigramTagger, BigramTagger, TrigramTagger
- NE chunking with regexp patterns
- Sentence boundary disambiguation (Punkt)
- Tree data structure with bracket parsing
- Full tokenizer suite: Treebank, Tweet, Regexp, Whitespace, WordPunct

### Changed
- Porter stemmer: pure Rust rewrite from NLTK reference
- Snowball stemmer: wraps rust-stemmers crate

### Fixed
- Decimal separator handling in Treebank tokenizer
- Contraction splitting edge cases

## [0.1.0] — 2025-03-15

### Added
- Initial release
- Word tokenization (TreebankWordTokenizer) — 94× speedup
- Sentence tokenization (PunktSentenceTokenizer)
- Porter stemming — 23× speedup
- Snowball stemming (English) — 8.8× speedup
- Lancaster stemming
- Regexp tokenization with fast-path
- Drop-in API: `from fastnltk import word_tokenize, sent_tokenize, pos_tag`
- Pre-built wheels: Linux (x86_64, aarch64), macOS (x86_64, arm64), Windows (x64)
- Python 3.8+ support via PyO3 abi3
- NLTK data compatibility (uses same corpus files)
- CI: GitHub Actions with lint, test, and PyPI release workflows
