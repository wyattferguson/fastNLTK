# Benchmarks

> **Last updated:** 2025-07-15 (v0.4.0, release build, commit `96e179d`)
> **Geometric mean: 7.6× vs NLTK** across 44 compared benchmarks (61 total).
>
> Run benchmarks: `python -m benchmarks.run --save`
> Fixtures: NLTK Gutenberg corpus (~200KB medium, ~5KB tiny).

---

## Highlights

| Operation | NLTK (ms) | fastNLTK (ms) | Speedup | Notes |
|---|---|---|---|---|
| **windowdiff** | 2.47 | **0.01** | **168×** | Pure algorithmic port, zero Python overhead |
| **edit_distance** | 2.44 | **0.02** | **152×** | Damerau-Levenshtein in Rust |
| **pk** | 2.20 | **0.02** | **93×** | Segmentation metric in Rust |
| **TreebankWordDetokenizer** | 6.96 | **0.20** | **35×** | Single-pass undo |
| **PunktSentenceTokenizer** | 14.56 | **0.43** | **34×** | Byte-level sentence scan |
| **Expression.fromstring** | 17.45 | **0.58** | **30×** | FOL parser in Rust |
| **SExprTokenizer** | 0.34 | **0.01** | **29×** | S-expression splitter |
| **PerceptronTagger.tag** | 17.64 | **0.66** | **27×** | u64 feature IDs, FxHashMap |
| **CFG.from_string** | 0.05 | **0.002** | **26×** | Grammar parser in Rust |
| **TweetTokenizer** | 85.44 | **3.51** | **24×** | LazyLock regexes |
| **EarleyChartParser.parse** | 6.65 | **0.32** | **21×** | Chart parser in Rust |
| **LancasterStemmer** | 33.86 | **2.01** | **17×** | Full 124-rule NLTK port |
| **TreebankWordTokenizer** | 42.56 | **2.54** | **17×** | Single-pass char scanner + SIMD |
| **QuadgramCollocationFinder** | 95.44 | **5.73** | **17×** | FastMap ngram counting |
| **TrigramCollocationFinder** | 56.01 | **4.06** | **14×** | FastMap ngram counting |
| **BlanklineTokenizer** | 2.24 | **0.17** | **13×** | Char scanner |
| **SnowballStemmer** | 22.18 | **1.91** | **12×** | rust-stemmers crate |

---

## Full Results (61 benchmarks)

Benchmarks grouped by module. Numbers from `python -m benchmarks.run --save` on release build.

| Module | Benchmark | NLTK (ms) | fastNLTK (ms) | Speedup |
|--------|-----------|-----------|---------------|---------|
| **tokenize** | | | | |
| | ToktokTokenizer.tokenize | 18.71 | 4.75 | **3.9×** |
| | MWETokenizer.tokenize | 1.17 | 1.19 | 1.0× |
| | RegexpTokenizer.tokenize | 4.61 | 3.86 | 1.2× |
| | SpaceTokenizer.tokenize | 1.06 | 2.70 | 0.4× |
| | TreebankWordTokenizer.tokenize | 42.56 | 2.54 | **16.8×** |
| | TweetTokenizer.tokenize | 85.44 | 3.51 | **24.3×** |
| | SExprTokenizer.tokenize | 0.34 | 0.01 | **28.7×** |
| | PunktSentenceTokenizer.tokenize | 14.56 | 0.43 | **34.0×** |
| | TreebankWordDetokenizer.detokenize | 6.96 | 0.20 | **34.7×** |
| | TabTokenizer.tokenize | 0.08 | 0.03 | **2.9×** |
| | LineTokenizer.tokenize | 0.24 | 0.16 | 1.5× |
| | WhitespaceTokenizer.tokenize | 4.50 | 1.52 | **3.0×** |
| | WordPunctTokenizer.tokenize | 5.85 | 1.78 | **3.3×** |
| | BlanklineTokenizer.tokenize | 2.24 | 0.17 | **13.1×** |
| | logos_word_tokenize † | — | 1.85 | — |
| **stem** | | | | |
| | SnowballStemmer.stem | 22.18 | 1.91 | **11.6×** |
| | PorterStemmer.stem | 43.97 | 6.85 | **6.4×** |
| | LancasterStemmer.stem | 33.86 | 2.01 | **16.9×** |
| | ARLSTem.stem | 1.70 | 0.45 | **3.8×** |
| | ISRIStemmer.stem | 2.03 | 0.28 | **7.3×** |
| | RSLPStemmer.stem † | — | 0.18 | — |
| | RegexpStemmer.stem † | — | 0.43 | — |
| **tag** | | | | |
| | PerceptronTagger.tag | 17.64 | 0.66 | **26.7×** |
| | TnT.tag | 1.05 | 0.21 | **4.9×** |
| | DefaultTagger.tag | 1.31 | 1.11 | 1.2× |
| | UnigramTagger.tag | 1.67 | 1.01 | **1.7×** |
| | BigramTagger.tag | 2.92 | 0.99 | **2.9×** |
| | TrigramTagger.tag | 3.06 | 1.04 | **2.9×** |
| | RegexpTagger.tag | 10.51 | 1.15 | **9.1×** |
| **classify** | | | | |
| | NaiveBayesClassifier.train | 6.61 | 1.95 | **3.4×** |
| | NaiveBayesClassifier.classify | 0.01 | 0.00 | **8.3×** |
| | TextCat.guess_language † | — | 4.20 | — |
| **probability** | | | | |
| | FreqDist.update | 19.40 | 4.98 | **3.9×** |
| | ConditionalFreqDist.inc | 5.04 | 2.64 | **1.9×** |
| | LaplaceProbDist.prob † | — | 0.0004 | — |
| | MLEProbDist.prob † | — | 0.0004 | — |
| **collocations** | | | | |
| | BigramCollocationFinder.from_words | 62.93 | 7.30 | **8.6×** |
| | TrigramCollocationFinder.from_words | 56.01 | 4.06 | **13.8×** |
| | QuadgramCollocationFinder.from_words | 95.44 | 5.73 | **16.7×** |
| **metrics** | | | | |
| | windowdiff | 2.47 | 0.01 | **168.3×** |
| | pk | 2.20 | 0.02 | **92.5×** |
| | edit_distance | 2.44 | 0.02 | **151.8×** |
| | BigramAssocMeasures † | — | 0.0003 | — |
| **lm** | | | | |
| | MLE.score † | — | 0.23 | — |
| | Lidstone.score † | — | 0.21 | — |
| | Laplace.score † | — | 0.20 | — |
| | StupidBackoff.score † | — | 0.16 | — |
| | KneserNeyInterpolated.score † | — | 0.17 | — |
| | WittenBellInterpolated.score † | — | 0.18 | — |
| **ccg** | | | | |
| | CCG from_string | 0.76 | 0.37 | **2.1×** |
| **chunk** | | | | |
| | RegexpParser.parse | 1.82 | 0.23 | **7.9×** |
| **parse** | | | | |
| | EarleyChartParser.parse | 6.65 | 0.32 | **21.0×** |
| | CFG.from_string | 0.05 | 0.002 | **26.3×** |
| **translate** | | | | |
| | bleu | 0.03 | 0.004 | **8.4×** |
| **chat** | | | | |
| | Chat.respond | 0.00 | 0.001 | **3.0×** |
| **tree** | | | | |
| | Tree.from_string | 3.23 | 0.33 | **9.8×** |
| **sem** | | | | |
| | Expression.fromstring | 17.45 | 0.58 | **30.0×** |
| **inference** | | | | |
| | TableauProver.prove † | — | 0.0006 | — |
| | ResolutionProver.prove † | — | 0.0007 | — |
| | DiscourseThread.answer_question † | — | 0.0018 | — |
| | DefaultReasoner.extensions † | — | 4.39 | — |

† fastNLTK-only — no NLTK comparison available.

---

## Module Leaderboard

| Module | Geo Mean Speedup | Best Single | Key Engine |
|--------|-----------------|-------------|------------|
| metrics | **133.2×** | 168× (windowdiff) | Pure algorithmic port, zero Python overhead |
| parse | **23.5×** | 26× (CFG) | Earley + CFG parsing |
| sem | **30.0×** | 30× (Expression) | FOL expression parser |
| collocations | **12.6×** | 17× (Quadgram) | FastMap ngram frequency counting |
| tree | **9.8×** | 10× | Tree bracket parser |
| translate | **8.4×** | 8× (BLEU) | BLEU in Rust |
| stem | **8.1×** | 17× (Lancaster) | 124-rule NLTK port |
| chunk | **7.9×** | 8× | Regexp chunk parser |
| tokenize | **5.3×** | 35× (Detokenizer) | SIMD memchr3 + char scanner |
| classify | **5.3×** | 8× (NaiveBayes) | Maxent GIS training |
| tag | **4.1×** | 27× (Perceptron) | u64 feature IDs, integer Viterbi |
| probability | **2.7×** | 4× (FreqDist) | FreqDist/ConditionalFreqDist |
| ccg | **2.1×** | 2× | CCG category parsing |
| chat | **3.0×** | 3× | Eliza chatbot |

---

## v0.4.0 Changes Since Last Benchmark

| Change | Before | After | Impact |
|--------|--------|-------|--------|
| **Lancaster stemmer** | 24 hand-picked rules | 124-rule NLTK compatible port | Full NLTK parity, 16.9× speedup |
| **TnT tagger** | Delegated to NLTK (0.8×) | Integer-ID Viterbi in Rust (4.9×) | **6× improvement** |
| **SpaceTokenizer** | Delegated to NLTK | Rust regex-based impl | Self-contained, no NLTK call |
| **data.find()** | NLTK-only path resolution | Rust path search with NLTK fallback | Faster cold start |
| **Tree.append()** | Missing | Rust-backed append (str/Tree children) | Enables chunk.py Tree usage |
| **Vendor removed** | 60+ rustling files, 10 transitive deps | Self-contained HMM + LM implementations | -10MB, 0 clippy warnings |
| **Test coverage** | 291 Rust / 249 Python | 312 Rust / 249 Python | +21 edge case tests |

---

## Build System

| Change | Before | After | Gain |
|--------|--------|-------|------|
| **Vendor (rustling) removed** | 60 files, 10 deps | 0 | -10MB artifacts, 0 clippy warnings |
| **sccache CI** | No cache reuse | sccache for all rustc | -30% repeated builds |
| **cargo-nextest** | Sequential tests | Parallel test execution | -40% test time |
| **Parallel codegen** | codegen-units=1 | codegen-units=256 (dev) | -20% check time |
| **`.cargo/config.toml`** | — | mold/lld docs, parallel profiles | Faster local builds |
