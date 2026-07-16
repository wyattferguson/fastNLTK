# Benchmarks

> **Last updated:** 2025-07-16 (v0.4.0, Rust 1.97.1, release build)
> **Geometric mean: 8.4× vs NLTK** across 44 compared benchmarks (68 total).
>
> Run benchmarks: `python -m benchmarks.run --save`
> Fixtures: NLTK Gutenberg corpus (~200KB medium, ~5KB tiny).

---

## Highlights

| Operation | NLTK (ms) | fastNLTK (ms) | Speedup | Notes |
|---|---|---|---|---|
| **edit_distance** | 2.48 | **0.01** | **182×** | Damerau-Levenshtein in Rust |
| **windowdiff** | 2.33 | **0.01** | **158×** | Pure algorithmic port, zero Python overhead |
| **pk** | 2.16 | **0.02** | **88×** | Segmentation metric in Rust |
| **SExprTokenizer** | 0.47 | **0.01** | **41×** | S-expression splitter |
| **PunktSentenceTokenizer** | 14.43 | **0.43** | **34×** | Byte-level sentence scan |
| **TreebankWordDetokenizer** | 6.66 | **0.20** | **33×** | Single-pass undo |
| **Expression.fromstring** | 16.50 | **0.57** | **29×** | FOL parser in Rust |
| **CFG.from_string** | 0.06 | **0.002** | **27×** | Grammar parser in Rust |
| **TweetTokenizer** | 83.49 | **3.57** | **23×** | LazyLock regexes |
| **LancasterStemmer** | 39.75 | **1.87** | **21×** | Full 124-rule NLTK port |
| **QuadgramCollocationFinder** | 106.84 | **5.29** | **20×** | FastMap ngram counting |
| **EarleyChartParser.parse** | 6.67 | **0.49** | **14×** | Chart parser in Rust |
| **TrigramCollocationFinder** | 59.74 | **4.13** | **15×** | FastMap ngram counting |
| **WordNetLemmatizer** | 6.23 | **0.48** | **13×** | Morphological rules in Rust |
| **SnowballStemmer** | 22.42 | **1.91** | **12×** | rust-stemmers crate |
| **Tree.from_string** | 3.82 | **0.32** | **12×** | Bracket parser in Rust |
| **TreebankWordTokenizer** | 42.36 | **4.05** | **10×** | Single-pass char scanner |

---

## Full Results (68 benchmarks)

Benchmarks grouped by module. Numbers from `python -m benchmarks.run --save` on release build, Rust 1.97.1.

| Module | Benchmark | NLTK (ms) | fastNLTK (ms) | Speedup |
|--------|-----------|-----------|---------------|---------|
| **tokenize** | | | | |
| | ToktokTokenizer.tokenize | 19.33 | 4.24 | **4.6×** |
| | MWETokenizer.tokenize | 1.16 | 1.17 | 1.0× |
| | RegexpTokenizer.tokenize | 6.54 | 3.73 | **1.8×** |
| | SpaceTokenizer.tokenize | 1.08 | 2.56 | 0.4× |
| | TreebankWordTokenizer.tokenize | 42.36 | 4.05 | **10.4×** |
| | TweetTokenizer.tokenize | 83.49 | 3.57 | **23.4×** |
| | SExprTokenizer.tokenize | 0.47 | 0.01 | **41.0×** |
| | PunktSentenceTokenizer.tokenize | 14.43 | 0.43 | **33.9×** |
| | TreebankWordDetokenizer.detokenize | 6.66 | 0.20 | **33.2×** |
| | TabTokenizer.tokenize | 0.08 | 0.03 | **3.0×** |
| | LineTokenizer.tokenize | 0.24 | 0.17 | 1.4× |
| | WhitespaceTokenizer.tokenize | 7.19 | 1.22 | **5.9×** |
| | WordPunctTokenizer.tokenize | 8.59 | 1.54 | **5.6×** |
| | BlanklineTokenizer.tokenize | 2.23 | 0.12 | **18.4×** |
| | logos_word_tokenize † | — | 1.52 | — |
| **stem** | | | | |
| | SnowballStemmer.stem | 22.42 | 1.91 | **11.7×** |
| | PorterStemmer.stem | 42.87 | 6.63 | **6.5×** |
| | LancasterStemmer.stem | 39.75 | 1.87 | **21.3×** |
| | WordNetLemmatizer.lemmatize | 6.23 | 0.48 | **13.0×** |
| | ARLSTem.stem | 1.63 | 0.44 | **3.7×** |
| | ISRIStemmer.stem | 2.13 | 0.27 | **7.9×** |
| | RSLPStemmer.stem † | — | 0.18 | — |
| | RegexpStemmer.stem † | — | 0.43 | — |
| **tag** | | | | |
| | PerceptronTagger.tag | 16.48 | 2.47 | **6.7×** |
| | TnT.tag | 1.02 | 0.22 | **4.7×** |
| | DefaultTagger.tag | 1.27 | 1.19 | 1.1× |
| | UnigramTagger.tag | 1.69 | 1.04 | **1.6×** |
| | BigramTagger.tag | 2.75 | 1.03 | **2.7×** |
| | TrigramTagger.tag | 2.98 | 1.08 | **2.8×** |
| | RegexpTagger.tag | 11.37 | 1.23 | **9.2×** |
| **classify** | | | | |
| | NaiveBayesClassifier.train | 6.32 | 2.04 | **3.1×** |
| | NaiveBayesClassifier.classify | 0.01 | 0.00 | **7.7×** |
| | TextCat.guess_language † | — | 4.19 | — |
| **probability** | | | | |
| | FreqDist.update | 20.26 | 4.93 | **4.1×** |
| | ConditionalFreqDist.inc | 5.10 | 2.62 | **2.0×** |
| | LaplaceProbDist.prob † | — | 0.0004 | — |
| | MLEProbDist.prob † | — | 0.0004 | — |
| **collocations** | | | | |
| | BigramCollocationFinder.from_words | 62.75 | 7.37 | **8.5×** |
| | TrigramCollocationFinder.from_words | 59.74 | 4.13 | **14.5×** |
| | QuadgramCollocationFinder.from_words | 106.84 | 5.29 | **20.2×** |
| **sentiment** | | | | |
| | SentimentIntensityAnalyzer.polarity_scores | 67.18 | 184.99 | 0.4× |
| **metrics** | | | | |
| | windowdiff | 2.33 | 0.01 | **158.3×** |
| | pk | 2.16 | 0.02 | **88.4×** |
| | edit_distance | 2.48 | 0.01 | **182.1×** |
| | BigramAssocMeasures † | — | 0.0003 | — |
| **lm** | | | | |
| | MLE.score † | — | 0.22 | — |
| | Lidstone.score † | — | 0.20 | — |
| | Laplace.score † | — | 0.20 | — |
| | StupidBackoff.score † | — | 0.16 | — |
| | KneserNeyInterpolated.score † | — | 0.18 | — |
| | WittenBellInterpolated.score † | — | 0.18 | — |
| **ccg** | | | | |
| | CCG from_string | 0.78 | 0.36 | **2.1×** |
| **chunk** | | | | |
| | RegexpParser.parse | 1.83 | 0.23 | **7.8×** |
| **parse** | | | | |
| | EarleyChartParser.parse | 6.67 | 0.49 | **13.6×** |
| | CFG.from_string | 0.06 | 0.002 | **27.4×** |
| **translate** | | | | |
| | bleu | 0.03 | 0.004 | **9.0×** |
| **chat** | | | | |
| | Chat.respond | 0.001 | 0.0003 | **3.5×** |
| **tree** | | | | |
| | Tree.from_string | 3.82 | 0.32 | **11.8×** |
| **sem** | | | | |
| | Expression.fromstring | 16.50 | 0.57 | **29.1×** |
| **inference** | | | | |
| | TableauProver.prove † | — | 0.0005 | — |
| | ResolutionProver.prove † | — | 0.0006 | — |
| | DiscourseThread.answer_question † | — | 0.0017 | — |
| | DefaultReasoner.extensions † | — | 4.45 | — |

† fastNLTK-only — no NLTK comparison available.

---

## Module Leaderboard

| Module | Geo Mean Speedup | Best Single | Key Engine |
|--------|-----------------|-------------|------------|
| metrics | **140.0×** | 182× (edit_distance) | Pure algorithmic port, zero Python overhead |
| parse | **19.3×** | 27× (CFG) | Earley + CFG parsing |
| sem | **29.1×** | 29× (Expression) | FOL expression parser |
| collocations | **13.5×** | 20× (Quadgram) | FastMap ngram frequency counting |
| tree | **11.8×** | 12× | Tree bracket parser |
| translate | **9.0×** | 9× (BLEU) | BLEU in Rust |
| stem | **8.3×** | 21× (Lancaster) | 124-rule NLTK port |
| chunk | **7.8×** | 8× | Regexp chunk parser |
| tokenize | **5.1×** | 41× (SExpr) | SIMD memchr3 + char scanner |
| classify | **4.9×** | 8× (NaiveBayes) | Maxent GIS training |
| tag | **3.2×** | 9× (RegexpTagger) | u64 feature IDs, integer Viterbi |
| probability | **2.9×** | 4× (FreqDist) | FreqDist/ConditionalFreqDist |
| ccg | **2.1×** | 2× | CCG category parsing |
| chat | **3.5×** | 4× | Eliza chatbot |
