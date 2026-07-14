# Benchmarks

> **Last updated:** 2026-07-14 (release build, i7-12700, 32GB RAM)
> **v0.3.0:** pyo3 v0.29, hashbrown v0.17, logos v0.16, phf v0.14, rand v0.10, smol_str v0.3, whatlang v0.18
> 279 Rust tests, 0 clippy errors, all deps at latest
>
> Times are **median** of 5–100 iterations. All benchmarks include NLTK comparison
> unless noted below.
>
> Run yourself: `.venv\Scripts\python -m benchmarks.run`

---

## All Benchmarks

| Module | Function | Input | NLTK (ms) | fastNLTK (ms) | Speedup |
|---|---|---|---|---|---|
| **tokenize** | | | | | |
| | `ToktokTokenizer.tokenize` | 82KB | 16.21 | 4.53 | **3.6x** |
| | `MWETokenizer.tokenize` | 13.6K words | 1.73 | 1.48 | 1.2x |
| | `RegexpTokenizer.tokenize` | 82KB | 3.80 | 2.68 | 1.4x |
| | `SpaceTokenizer.tokenize` | 82KB | 0.74 | 0.85 | 0.9x |
| | `TreebankWordTokenizer.tokenize` | 82KB | 43.34 | 2.71 | **16.0x** |
| | `TweetTokenizer.tokenize` | 82KB | 100.61 | 7.10 | **14.2x** |
| | `TextTilingTokenizer.tokenize` | 82KB | 6.15 | 0.08 | **76.7x** |
| | `logos_word_tokenize` 🆕 | 82KB | — | 1.20 | —¹ |
| **stem** | | | | | |
| | `SnowballStemmer.stem` | 10K words | 78.94 | 4.16 | **19.0x** |
| | `PorterStemmer.stem` | 10K words | 144.73 | 17.41 | **8.3x** |
| | `LancasterStemmer.stem` | 10K words | 72.71 | 3.33 | **21.9x** |
| | `WordNetLemmatizer.lemmatize` | 5K words | 9.93 | 0.58 | **17.2x** |
| **tag** | | | | | |
| | `PerceptronTagger.tag` | 100 sentences | 35.30 | 10.02 | **3.5x** |
| | `HiddenMarkovModelTagger.tag` | 1K words | — | 0.59 | —² |
| | `DefaultTagger.tag` | 10K words | 2.91 | 2.42 | 1.2x |
| | `UnigramTagger.tag` | 10K words | 3.87 | 2.37 | **1.6x** |
| | `BigramTagger.tag` | 10K words | 6.68 | 2.92 | **2.3x** |
| | `TrigramTagger.tag` | 10K words | 7.00 | 3.38 | **2.1x** |
| | `RegexpTagger.tag` | 10K words | 25.48 | 2.51 | **10.2x** |
| | `AffixTagger.tag` | 10K words | 5.61 | 2.79 | **2.0x** |
| **classify** | | | | | |
| | `NaiveBayesClassifier.train` | 2K instances | 17.10 | 4.91 | **3.5x** |
| | `NaiveBayesClassifier.classify` | 5 features | 0.01 | 0.00 | **8.3x** |
| **probability** | | | | | |
| | `FreqDist.update` | 100K samples | 33.93 | 5.34 | **6.3x** |
| **collocations** | | | | | |
| | `BigramCollocationFinder.from_words` | 50K words | 85.15 | 9.06 | **9.4x** |
| **sentiment** | | | | | |
| | `SentimentIntensityAnalyzer.polarity_scores` | 82KB | 44.34 | 0.97 | **45.9x** |
| **metrics** | | | | | |
| | `windowdiff` | 12K chars | 5.12 | 0.04 | **113.9x** |
| | `pk` | 12K chars | 4.75 | 0.08 | **56.0x** |
| | `edit_distance` | 100 chars | 6.15 | 0.04 | **175.1x** |
| **lm** | | | | | |
| | `MLE.fit` | 1K sentences | — | 1.37 | —³ |
| | `KneserNeyInterpolated.score` | 4 queries | — | 0.006 | —³ |
| **ccg** | | | | | |
| | `CCG from_string` | 3.5K parses | 2.16 | 1.13 | **1.9x** |
| **chunk** | | | | | |
| | `RegexpParser.parse` | 1.8K tokens | 3.39 | 0.77 | **4.4x** |
| **cluster** | | | | | |
| | `KMeansClusterer.cluster` | 500×5D | 3.92 | 0.91 | **4.3x** |
| **parse** | | | | | |
| | `EarleyChartParser.parse` | 30 sentences | 22.41 | 0.83 | **27.1x** |
| **translate** | | | | | |
| | `bleu` | 7 tokens | 0.08 | 0.01 | **9.7x** |
| **chat** | | | | | |
| | `Chat.respond` | single | 0.003 | 0.001 | **3.5x** |
| **tree** | | | | | |
| | `Tree.from_string` | 300 trees | 8.84 | 0.82 | **10.8x** |
| **sem** | | | | | |
| | `Expression.fromstring` | 500 formulas | 53.76 | 1.40 | **38.4x** |
| **inference** | | | | | |
| | `TableauProver.prove` | P\|~P | — | 0.001 | —⁴ |
| | `ResolutionProver.prove` | P\|~P | — | 0.001 | —⁴ |
| | `DiscourseThread.answer_question` | 2 DRSs | — | 0.004 | —⁴ |
| | `DefaultReasoner.extensions` | 10 rules | — | 58.70 | —⁴ |
| **Average (42 benchmarks)** | | | | | **14.3x** |

**Footnotes:**
- ¹ 🆕 **fastNLTK-exclusive** — DFA lexer via `logos` crate, no NLTK equivalent
- ² fastNLTK-only — NLTK has `nltk.tag.hmm` but with a different API (not directly comparable)
- ³ Exists in NLTK but NLTK's LM API (`nltk.lm`) is version-incompatible with the test data format
- ⁴ Exists in NLTK but NLTK's inference API has bugs with these formulas (`AttributeError`, `skolemize` errors)

---

## Top 10 Speedups

| # | Function | Speedup | Why |
|---|---|---|---|
| 1 | `edit_distance` | **175.1x** | DP in native code vs Python loop |
| 2 | `windowdiff` | **113.9x** | Pure algorithmic port, no Python loop overhead |
| 3 | `TextTilingTokenizer.tokenize` | **76.7x** | Algorithmic port, zero Python overhead |
| 4 | `pk` | **56.0x** | Same as windowdiff — simple string scan |
| 5 | `SentimentIntensityAnalyzer.polarity_scores` | **45.9x** | VADER algorithm in Rust vs Python |
| 6 | `Expression.fromstring` | **38.4x** | Recursive descent parser in native code |
| 7 | `EarleyChartParser.parse` | **27.1x** | Chart parsing in Rust vs Python |
| 8 | `LancasterStemmer.stem` | **21.9x** | Algorithmic port, string ops in native code |
| 9 | `SnowballStemmer.stem` | **19.0x** | `rust-stemmers` — libstemmer in Rust |
| 10 | `WordNetLemmatizer.lemmatize` | **17.2x** | Dictionary lookup in Rust vs Python |

---

## Module Coverage

| Module | Benchmarks | Best Speedup |
|---|---|---|
| tokenize | 8 | **76.7x** |
| stem | 4 | **21.9x** |
| tag | 8 | **10.2x** |
| classify | 2 | **8.3x** |
| probability | 1 | **6.3x** |
| collocations | 1 | **9.4x** |
| sentiment | 1 | **45.9x** |
| metrics | 3 | **175.1x** |
| lm | 2 | fastNLTK-only |
| ccg | 1 | **1.9x** |
| chunk | 1 | **4.4x** |
| cluster | 1 | **4.3x** |
| parse | 1 | **27.1x** |
| translate | 1 | **9.7x** |
| chat | 1 | **3.5x** |
| tree | 1 | **10.8x** |
| sem | 1 | **38.4x** |
| inference | 4 | fastNLTK-only |

---

## Running

```bash
.venv\Scripts\python -m benchmarks.run           # Run all
.venv\Scripts\python -m benchmarks.run --save    # Run + save
```

The harness (42 automated benchmarks in `benchmarks/bench_suite.py`) supports automatic
regression detection against saved baselines. Default threshold: 25%.
