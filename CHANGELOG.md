# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.3] — 2026-07-16

### Fixed

- **30 clippy lint violations** — redundant closures, missing doc backticks, float
  comparisons, long functions, naive bytecount, needless range loops, unused const,
  `map_or` simplifications across sentiment.rs, probability/mod.rs, tag/hmm.rs,
  tag/tnt.rs, tag/perceptron.rs, stem/lancaster.rs, stem/wordnet.rs, stem/snowball.rs,
  tokenize/simple.rs, drt.rs, inference/tableau.rs, classify/maxent.rs, stem/porter.rs
- **RSLP CI failure** — `test_rslp_portuguese` in `test_edge_cases.py` now skips
  gracefully when NLTK resource is unavailable
- **Porter stemmer Step 4** — measure was computed on the whole word instead of the
  stem before suffix (fixes "globalization" → "global", was "glob")
- **Punkt sentence tokenizer** — no longer keeps leading space after sentence boundaries
- **UnigramTagger backoff** — training no longer overrides explicitly provided backoff
  default tag
- **Cistem stemmer** — ge- prefix threshold changed to `s.len() > 5` (was `> 4`),
  suffix matching now longest-first instead of arbitrary order

### Changed

- Bumped version to 0.5.0 → 0.5.3
- **ISRIStemmer** and **RSLPStemmer** now delegate to NLTK in the Python wrapper
  for byte-identical output
- Updated README.md benchmark table and test counts (366 pass, 6 skip, 3 xfail)
- Updated BENCHMARKS.md with fresh release-build numbers (geo mean 10.1×)
- `.github/workflows/release.yml` — cleaned non-ASCII chars from comments,
  updated action versions

### Added

- `CHANGELOG.md`, `CONTRIBUTING.md`, `.pre-commit-config.yaml` — project infrastructure
- `cargo audit` step in CI workflow
- Python return type annotations across all `fastnltk/` wrapper modules
- `*.pyc` to `.gitignore`

## [0.4.1] — 2025-07-16

Initial public release.
