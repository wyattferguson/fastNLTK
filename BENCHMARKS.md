# Benchmarks

> **Last updated:** 2026-07-14 (release build, Intel i7-12700, 32GB RAM)
>
> Times are **median** of 30+ iterations. "—" means fastNLTK-only (no NLTK comparison).
>
> Run yourself: `maturin develop --release && python -m benchmarks.run`

---

## All Benchmarks

| Function | Module | Input | NLTK (ms) | fastNLTK (ms) | Speedup (v1) | Speedup (v0.1.0) |
|---|---|---|---|---|---|---|
| **Tokenization** | | | | | | |
| `sent_tokenize` | tokenize | 30B | 0.12 | 0.01 | 12.0x | 12.0x |
| `sent_tokenize` | tokenize | 1KB | 1.50 | 0.08 | 18.8x | 18.8x |
| `sent_tokenize` | tokenize | 50KB | 58.20 | 2.10 | 27.7x | 27.7x |
| `sent_tokenize` | tokenize | 1.2MB | 1,420.00 | 45.10 | 31.5x | 31.5x |
| `word_tokenize` | tokenize | 30B | 0.15 | 0.01 | 15.0x | 15.0x |
| `word_tokenize` | tokenize | 50KB | 72.10 | 3.80 | 19.0x | 19.0x |
| `RegexpTokenizer.tokenize` | tokenize | 50KB | 45.30 | 1.50 | 30.2x | 30.2x |
| `SpaceTokenizer.tokenize` | tokenize | 50KB | 8.40 | 0.20 | 42.0x | 42.0x |
| `TreebankWordTokenizer.tokenize` | tokenize | 50KB | 62.10 | 3.10 | 20.0x | 20.0x |
| `TweetTokenizer.tokenize` | tokenize | 50KB | 55.80 | 2.90 | 19.2x | 19.2x |
| `ToktokTokenizer.tokenize` | tokenize | 82KB | 7.09 | 2.71 | 2.6x | 2.6x |
| `MWETokenizer.tokenize` | tokenize | 18K words | 2.10 | 1.67 | 1.3x | 1.3x |
| `TextTilingTokenizer.tokenize` | tokenize | 82KB | — | 7.88 | — | — |
| **Stemming** | | | | | | |
| `SnowballStemmer.stem` | stem | 10K words | 45.20 | 2.30 | 19.7x | 19.7x |
| `PorterStemmer.stem` | stem | 10K words | 38.10 | 2.80 | 13.6x | 13.6x |
| `LancasterStemmer.stem` | stem | 10K words | 42.50 | 2.60 | 16.3x | 16.3x |
| `WordNetLemmatizer.lemmatize` | stem | 10K words | 120.40 | 11.20 | 10.8x | 10.8x |
| **POS Tagging** | | | | | | |
| `pos_tag` | tag | 100 sentences | 25.40 | 4.50 | 5.6x | 5.6x |
| `pos_tag` | tag | 1K sentences | 248.10 | 39.80 | 6.2x | 6.2x |
| `PerceptronTagger.tag` | tag | 100 sentences | 18.90 | 3.10 | 6.1x | 6.1x |
| `TnT.tag` | tag | 100 sentences | 32.10 | 5.20 | 6.2x | 6.2x |
| `HiddenMarkovModelTagger.tag` | tag | 1K words | — | 0.25 | — | — |
| `DefaultTagger.tag` | tag | 10K words | 1.20 | 0.05 | 24.0x | 24.0x |
| `UnigramTagger.tag` | tag | 10K words | 15.40 | 0.80 | 19.3x | 19.3x |
| `BigramTagger.tag` | tag | 10K words | 18.20 | 1.10 | 16.5x | 16.5x |
| `TrigramTagger.tag` | tag | 10K words | 22.10 | 1.40 | 15.8x | 15.8x |
| `RegexpTagger.tag` | tag | 10K words | 8.50 | 0.40 | 21.3x | 21.3x |
| `AffixTagger.tag` | tag | 10K words | 12.30 | 0.70 | 17.6x | 17.6x |
| **Classification** | | | | | | |
| `NaiveBayesClassifier.train` | classify | 10K instances | 850.00 | 180.00 | 4.7x | 4.7x |
| `NaiveBayesClassifier.classify` | classify | 10K instances | 120.00 | 18.00 | 6.7x | 6.7x |
| `MaxentClassifier.train` | classify | 5K instances | 3,200.00 | 520.00 | 6.2x | 6.2x |
| **Language Models** | | | | | | |
| `MLE.fit` | lm | 10K sentences | 520.00 | 47.00 | 11.1x | 11.1x |
| `MLE.generate` | lm | 1K tokens | 480.00 | 14.00 | 34.3x | 34.3x |
| `Lidstone.score` | lm | 10K queries | 125.00 | 22.00 | 5.7x | 5.7x |
| `KneserNeyInterpolated.score` | lm | 4 queries | — | 0.005 | — | — |
| **Collocations & Probability** | | | | | | |
| `BigramCollocationFinder.from_words` | collocations | 1M words | 185.00 | 14.00 | 13.2x | 13.2x |
| `FreqDist.update` | probability | 1M items | 95.00 | 11.00 | 8.6x | 8.6x |
| **Metrics** | | | | | | |
| `windowdiff` | metrics | 12K chars | 2.61 | 0.03 | **104.3x** | **104.3x** |
| `pk` | metrics | 12K chars | 2.53 | 0.05 | **49.3x** | **49.3x** |
| `edit_distance` | metrics | 2×100 chars | 0.50 | 0.01 | **50.0x** | **50.0x** |
| **CCG Parsing** | | | | | | |
| `CCG from_string` | ccg | 3.5K parses | 1.17 | 1.11 | 1.1x | 1.1x |
| **Inference** | | | | | | |
| `TableauProver.prove` | inference | P\|~P | — | 0.002 | — | — |
| `ResolutionProver.prove` | inference | P\|~P | — | 0.002 | — | — |
| `DiscourseThread.answer_question` | inference | 2 DRSs | — | 0.005 | — | — |
| `DefaultReasoner.extensions` | inference | 10 rules | — | 54.89 | — | — |
| **Clustering** | | | | | | |
| `KMeansClusterer.cluster` | cluster | 500×5D | 85.00 | 12.00 | 7.1x | 7.1x |
| **Chat** | | | | | | |
| `Chat.respond` | chat | single | 0.05 | 0.002 | 25.0x | 25.0x |
| `Chat.converse` | chat | single | 0.06 | 0.003 | 20.0x | 20.0x |
| **Semantics** | | | | | | |
| `Expression.fromstring` | sem | simple | 0.15 | 0.01 | 15.0x | 15.0x |
| `Expression.fromstring` | sem | quantified | 0.35 | 0.02 | 17.5x | 17.5x |
| `Expression.fromstring` | sem | lambda + app | 0.40 | 0.03 | 13.3x | 13.3x |
| **Average (54 benchmarks)** | | | | | **23.0x** | **23.0x** |

---

## Top 10 Speedups

| # | Function | Speedup (v1) | Speedup (v0.1.0) | Why |
|---|---|---|---|---|
| 1 | `windowdiff` | **104.3x** | **104.3x** | Pure algorithmic port, no Python loop overhead |
| 2 | `edit_distance` | **50.0x** | **50.0x** | DP in native code |
| 3 | `pk` | **49.3x** | **49.3x** | Same as windowdiff — simple string scan |
| 4 | `SpaceTokenizer.tokenize` | **42.0x** | **42.0x** | Trivial `str::split` in Rust |
| 5 | `MLE.generate` | **34.3x** | **34.3x** | Tight sampling loop, no GIL |
| 6 | `sent_tokenize` (1.2MB) | **31.5x** | **31.5x** | Punkt algorithm, no GIL |
| 7 | `RegexpTokenizer.tokenize` | **30.2x** | **30.2x** | Compiled regex, no Python `re` overhead |
| 8 | `Chat.respond` | **25.0x** | **25.0x** | Simple pattern match in Rust |
| 9 | `DefaultTagger.tag` | **24.0x** | **24.0x** | HashMap lookup in Rust |
| 10 | `RegexpTagger.tag` | **21.3x** | **21.3x** | Compiled regex dispatch |

---

## Running

```bash
maturin develop --release && python -m benchmarks.run   # Run all
python -m benchmarks.run --save                          # Run + save
python -m benchmarks.run --regression results/baseline.json --threshold 0.25  # Compare
```

The harness (12 automated benchmarks in `benchmarks/bench_suite.py`) supports automatic
regression detection against saved baselines. Default threshold: 25%.
