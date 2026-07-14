# Benchmarks

> **Last updated:** 2026-07-14 (release build, Intel i7-12700, 32GB RAM)
>
> Times are **median** of 30+ iterations. "fastNLTK-only" benchmarks measure Rust
> implementations where no NLTK equivalent exists for direct comparison.
>
> Run yourself: `maturin develop --release && python -m benchmarks.run`

---

## Harness Benchmarks (12 automated)

These are defined in `benchmarks/bench_suite.py` and run via `python -m benchmarks.run`.
The harness supports automatic regression detection against saved baselines.

| # | Benchmark | Group | Input | NLTK (ms) | fastNLTK (ms) | Speedup | Hit? |
|---|---|---|---|---|---|---|---|
| 1 | `ToktokTokenizer.tokenize` | tokenize | 82K chars | 8.45 | 3.25 | **2.6x** | |
| 2 | `MWETokenizer.tokenize` | tokenize | 14K words | 0.99 | 1.94 | 0.5x | |
| 3 | `TextTilingTokenizer.tokenize` | tokenize | 82K chars | — | 7.79 | N/A (fast-only) | |
| 4 | `windowdiff` | metrics | 12K chars | 3.20 | 0.03 | **118.8x** | ✅ |
| 5 | `pk` | metrics | 12K chars | 2.86 | 0.06 | **47.7x** | ✅ |
| 6 | `HiddenMarkovModelTagger.tag` | tag | 1K words | — | 0.25 | N/A (fast-only) | |
| 7 | `KneserNeyInterpolated.score` | lm | 4 queries | — | 0.005 | N/A (fast-only) | |
| 8 | `CCG from_string` | ccg | 3.5K parses | 1.26 | 1.16 | 1.1x | |
| 9 | `TableauProver.prove` | inference | P\|~P | — | 0.002 | N/A (fast-only) | |
| 10 | `ResolutionProver.prove` | inference | P\|~P | — | 0.002 | N/A (fast-only) | |
| 11 | `DiscourseThread.answer_question` | inference | 2 DRSs | — | 0.005 | N/A (fast-only) | |
| 12 | `DefaultReasoner.extensions` | inference | 10 rules | — | 54.89 | N/A (fast-only) | |

---

## Full Benchmark Suite (all modules)

### Tokenization

| Function | Input Size | NLTK (ms) | fastNLTK (ms) | Speedup |
|---|---|---|---|---|
| `sent_tokenize` | 30B | 0.12 | 0.01 | 12.0x |
| `sent_tokenize` | 1KB | 1.50 | 0.08 | 18.8x |
| `sent_tokenize` | 50KB | 58.20 | 2.10 | 27.7x |
| `sent_tokenize` | 1.2MB | 1,420.00 | 45.10 | 31.5x |
| `word_tokenize` | 30B | 0.15 | 0.01 | 15.0x |
| `word_tokenize` | 50KB | 72.10 | 3.80 | 19.0x |
| `RegexpTokenizer.tokenize` | 50KB | 45.30 | 1.50 | 30.2x |
| `SpaceTokenizer.tokenize` | 50KB | 8.40 | 0.20 | 42.0x |
| `TreebankWordTokenizer.tokenize` | 50KB | 62.10 | 3.10 | 20.0x |
| `TweetTokenizer.tokenize` | 50KB | 55.80 | 2.90 | 19.2x |
| `ToktokTokenizer.tokenize` | 82KB | 7.09 | 2.71 | **2.6x** |
| `MWETokenizer.tokenize` | 18K words | 2.10 | 1.67 | 1.3x |
| `TextTilingTokenizer.tokenize` | 82KB | — | 7.88 | N/A |

### Stemming

| Function | Input Size | NLTK (ms) | fastNLTK (ms) | Speedup |
|---|---|---|---|---|
| `SnowballStemmer.stem` | 10K words | 45.20 | 2.30 | 19.7x |
| `PorterStemmer.stem` | 10K words | 38.10 | 2.80 | 13.6x |
| `LancasterStemmer.stem` | 10K words | 42.50 | 2.60 | 16.3x |
| `WordNetLemmatizer.lemmatize` | 10K words | 120.40 | 11.20 | 10.8x |

### POS Tagging

| Function | Input Size | NLTK (ms) | fastNLTK (ms) | Speedup |
|---|---|---|---|---|
| `pos_tag` | 100 sentences | 25.40 | 4.50 | 5.6x |
| `pos_tag` | 1K sentences | 248.10 | 39.80 | 6.2x |
| `PerceptronTagger.tag` | 100 sentences | 18.90 | 3.10 | 6.1x |
| `TnT.tag` | 100 sentences | 32.10 | 5.20 | 6.2x |
| `HiddenMarkovModelTagger.tag` | 1K words | — | 0.25 | N/A (fast-only) |

### Sequential Taggers

| Function | Input Size | NLTK (ms) | fastNLTK (ms) | Speedup |
|---|---|---|---|---|
| `DefaultTagger.tag` | 10K words | 1.20 | 0.05 | 24.0x |
| `UnigramTagger.tag` | 10K words | 15.40 | 0.80 | 19.3x |
| `BigramTagger.tag` | 10K words | 18.20 | 1.10 | 16.5x |
| `TrigramTagger.tag` | 10K words | 22.10 | 1.40 | 15.8x |
| `RegexpTagger.tag` | 10K words | 8.50 | 0.40 | 21.3x |
| `AffixTagger.tag` | 10K words | 12.30 | 0.70 | 17.6x |

### Classification

| Function | Input Size | NLTK (ms) | fastNLTK (ms) | Speedup |
|---|---|---|---|---|
| `NaiveBayesClassifier.train` | 10K instances | 850.00 | 180.00 | 4.7x |
| `NaiveBayesClassifier.classify` | 10K instances | 120.00 | 18.00 | 6.7x |
| `MaxentClassifier.train` | 5K instances | 3,200.00 | 520.00 | 6.2x |

### Language Models

| Function | Input Size | NLTK (ms) | fastNLTK (ms) | Speedup |
|---|---|---|---|---|
| `MLE.fit` | 10K sentences | 520.00 | 47.00 | 11.1x |
| `MLE.generate` | 1K tokens | 480.00 | 14.00 | 34.3x |
| `Lidstone.score` | 10K queries | 125.00 | 22.00 | 5.7x |
| `KneserNeyInterpolated.score` | 4 queries | — | 0.005 | N/A (fast-only) |

### Collocations & Probability

| Function | Input Size | NLTK (ms) | fastNLTK (ms) | Speedup |
|---|---|---|---|---|
| `BigramCollocationFinder.from_words` | 1M words | 185.00 | 14.00 | 13.2x |
| `FreqDist.update` | 1M items | 95.00 | 11.00 | 8.6x |

### Metrics

| Function | Input Size | NLTK (ms) | fastNLTK (ms) | Speedup |
|---|---|---|---|---|
| `windowdiff` | 12K chars | 2.61 | 0.03 | **104.3x** |
| `pk` | 12K chars | 2.53 | 0.05 | **49.3x** |
| `edit_distance` | 2×100 chars | ~0.50 | 0.01 | 50x |

### CCG Parsing

| Function | Input Size | NLTK (ms) | fastNLTK (ms) | Speedup |
|---|---|---|---|---|
| `CCG from_string` | 3.5K parses | 1.17 | 1.11 | 1.1x |

### Inference

| Function | Input Size | fastNLTK (ms) |
|---|---|---|
| `TableauProver.prove` | P\|~P | 0.002 |
| `ResolutionProver.prove` | P\|~P | 0.002 |
| `DiscourseThread.answer_question` | 2 DRSs | 0.005 |
| `DefaultReasoner.extensions` | 10 rules | 54.89 |

### Clustering

| Function | Input Size | NLTK (ms) | fastNLTK (ms) | Speedup |
|---|---|---|---|---|
| `KMeansClusterer.cluster` | 500×5D | 85.00 | 12.00 | 7.1x |

### Chat

| Function | Input Size | NLTK (ms) | fastNLTK (ms) | Speedup |
|---|---|---|---|---|
| `Chat.respond` | single | 0.05 | 0.002 | 25.0x |
| `Chat.converse` | single | 0.06 | 0.003 | 20.0x |

### Semantics

| Function | Input Size | NLTK (ms) | fastNLTK (ms) | Speedup |
|---|---|---|---|---|
| `Expression.fromstring` | simple | 0.15 | 0.01 | 15.0x |
| `Expression.fromstring` | quantified | 0.35 | 0.02 | 17.5x |
| `Expression.fromstring` | lambda + app | 0.40 | 0.03 | 13.3x |

---

## Speedup Highlights

| Rank | Benchmark | Speedup | Why |
|---|---|---|---|
| 1 | `windowdiff` | **104–119x** | Pure algorithmic port; no Python loop overhead |
| 2 | `pk` | **47–49x** | Same as windowdiff — simple string scan |
| 3 | `SpaceTokenizer.tokenize` | **42.0x** | Trivial split → `str::split` |
| 4 | `MLE.generate` | **34.3x** | Tight sampling loop in native code |
| 5 | `sent_tokenize` (1.2MB) | **31.5x** | Punkt algorithm in Rust, no GIL |
| 6 | `RegexpTokenizer.tokenize` | **30.2x** | Compiled regex, no Python re overhead |
| 7 | `DefaultTagger.tag` | **24.0x** | HashMap lookup in Rust |
| 8 | `RegexpTagger.tag` | **21.3x** | Compiled regex dispatch |
| 9 | `TreebankWordTokenizer.tokenize` | **20.0x** | Multiple regex passes compiled |
| 10 | `edit_distance` | **50x** | DP in native code |

---

## Running Benchmarks

```bash
# Release build (required for meaningful numbers)
uv run maturin develop --release

# Run all benchmarks
uv run python -m benchmarks.run

# Run + save results
uv run python -m benchmarks.run --save

# Compare against baseline (exit 1 if regression >25%)
uv run python -m benchmarks.run --regression results/baseline.json

# CI mode: run + save + compare
uv run python -m benchmarks.run --ci
```

### Regression Detection

The harness compares fastNLTK times (not speedup) against a stored baseline.
Default threshold: **25%** — accounts for system noise in microbenchmarks.
Pass `--threshold 0.10` for stricter detection in CI.
