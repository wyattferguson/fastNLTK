# Benchmarks

> **Last updated:** 2025-07-16 (v0.4.1, Rust 1.97.1, release build)
> **Geometric mean: 8.3× vs NLTK** across 49 compared benchmarks (66 total).
>
> Run benchmarks: `python -m benchmarks.run --save`
> Fixtures: NLTK Gutenberg corpus (~200KB medium, ~5KB tiny).

---

## Highlights

| Operation | NLTK (ms) | fastNLTK (ms) | Speedup | Notes |
|---|---|---|---|---|
| **TextTilingTokenizer** | 22366.13 | **34.13** | **655×** | Sentence segmentation, SIMD-accelerated |
| **edit_distance** | 2.53 | **0.01** | **180×** | Damerau-Levenshtein in Rust |
| **windowdiff** | 2.42 | **0.02** | **160×** | Pure algorithmic port, zero Python overhead |
| **pk** | 2.20 | **0.02** | **91×** | Segmentation metric in Rust |
| **MaxentClassifier.train** | 45.02 | **0.09** | **495×** | GIS training, fully optimized inner loop |
| **SExprTokenizer** | 0.36 | **0.01** | **30×** | S-expression splitter |
| **PunktSentenceTokenizer** | 14.41 | **0.46** | **32×** | Byte-level sentence scan |
| **TreebankWordDetokenizer** | 6.85 | **0.21** | **33×** | Single-pass undo |
| **Expression.fromstring** | 17.45 | **0.60** | **29×** | FOL parser in Rust |
| **CFG.from_string** | 0.05 | **0.002** | **24×** | Grammar parser in Rust |
| **TweetTokenizer** | 86.61 | **3.29** | **26×** | LazyLock regexes |
| **LancasterStemmer** | 32.23 | **1.86** | **17×** | Full 124-rule NLTK port |
| **QuadgramCollocationFinder** | 98.92 | **5.44** | **18×** | FastMap ngram counting |
| **EarleyChartParser.parse** | 7.37 | **0.51** | **14×** | Chart parser in Rust |
| **TrigramCollocationFinder** | 58.37 | **4.31** | **14×** | FastMap ngram counting |
| **WordNetLemmatizer** | 6.36 | **0.48** | **13×** | Morphological rules in Rust |
| **SnowballStemmer** | 22.50 | **2.04** | **11×** | rust-stemmers crate |
| **Tree.from_string** | 3.34 | **0.33** | **10×** | Bracket parser in Rust |
| **TreebankWordTokenizer** | 42.52 | **4.09** | **10×** | Single-pass char scanner |

---

## Full Results (66 benchmarks)

Benchmarks grouped by module. Numbers from `python -m benchmarks.run --save` on release build, Rust 1.97.1.

| Module | Benchmark | NLTK (ms) | fastNLTK (ms) | Speedup |
|--------|-----------|-----------|---------------|---------|
| **tokenize** | | | | |
| | ToktokTokenizer.tokenize | 18.74 | 4.34 | **4.3×** |
| | MWETokenizer.tokenize | 1.15 | 1.19 | 1.0× |
| | RegexpTokenizer.tokenize | 4.43 | 3.84 | 1.2× |
| | SpaceTokenizer.tokenize | 1.01 | 2.64 | 0.4× |
| | TreebankWordTokenizer.tokenize | 42.52 | 4.09 | **10.4×** |
| | TweetTokenizer.tokenize | 86.61 | 3.29 | **26.3×** |
| | TextTilingTokenizer.tokenize | 22366.13 | 34.13 | **655.4×** |
| | SExprTokenizer.tokenize | 0.36 | 0.01 | **30.4×** |
| | PunktSentenceTokenizer.tokenize | 14.41 | 0.46 | **31.6×** |
| | TreebankWordDetokenizer.detokenize | 6.85 | 0.21 | **32.9×** |
| | TabTokenizer.tokenize | 0.08 | 0.03 | **3.0×** |
| | LineTokenizer.tokenize | 0.25 | 0.16 | 1.5× |
| | WhitespaceTokenizer.tokenize | 4.28 | 1.27 | **3.4×** |
| | WordPunctTokenizer.tokenize | 5.60 | 1.61 | **3.5×** |
| | BlanklineTokenizer.tokenize | 2.27 | 0.12 | **18.8×** |
| | logos_word_tokenize † | — | 1.55 | — |
| **stem** | | | | |
| | SnowballStemmer.stem | 22.50 | 2.04 | **11.0×** |
| | PorterStemmer.stem | 44.59 | 6.45 | **6.9×** |
| | LancasterStemmer.stem | 32.23 | 1.86 | **17.3×** |
| | WordNetLemmatizer.lemmatize | 6.36 | 0.48 | **13.2×** |
| | ARLSTem.stem | 1.69 | 0.44 | **3.8×** |
| | ISRIStemmer.stem | 2.06 | 0.27 | **7.5×** |
| | RSLPStemmer.stem † | — | 0.19 | — |
| | RegexpStemmer.stem † | — | 0.43 | — |
| **tag** | | | | |
| | PerceptronTagger.tag | 17.07 | 2.48 | **6.9×** |
| | TnT.tag | 1.02 | 0.21 | **4.9×** |
| | DefaultTagger.tag | 1.28 | 1.21 | 1.1× |
| | UnigramTagger.tag | 1.68 | 1.04 | **1.6×** |
| | BigramTagger.tag | 2.93 | 1.05 | **2.8×** |
| | TrigramTagger.tag | 2.95 | 1.10 | **2.7×** |
| | RegexpTagger.tag | 9.80 | 1.24 | **7.9×** |
| **classify** | | | | |
| | NaiveBayesClassifier.train | 6.40 | 2.07 | **3.1×** |
| | NaiveBayesClassifier.classify | 0.01 | 0.00 | **7.7×** |
| | MaxentClassifier.train | 45.02 | 0.09 | **494.7×** |
| | TextCat.guess_language † | — | 4.38 | — |
| **probability** | | | | |
| | FreqDist.update | 20.68 | 5.06 | **4.1×** |
| | ConditionalFreqDist.inc | 5.02 | 2.62 | **1.9×** |
| | LaplaceProbDist.prob † | — | 0.0004 | — |
| | MLEProbDist.prob † | — | 0.0004 | — |
| **collocations** | | | | |
| | BigramCollocationFinder.from_words | 62.16 | 7.46 | **8.3×** |
| | TrigramCollocationFinder.from_words | 58.37 | 4.31 | **13.5×** |
| | QuadgramCollocationFinder.from_words | 98.92 | 5.44 | **18.2×** |
| **sentiment** | | | | |
| | SentimentIntensityAnalyzer.polarity_scores | 76.58 | 223.14 | 0.3× |
| **metrics** | | | | |
| | windowdiff | 2.42 | 0.02 | **160.2×** |
| | pk | 2.20 | 0.02 | **91.2×** |
| | edit_distance | 2.53 | 0.01 | **179.6×** |
| | BigramAssocMeasures † | — | 0.0003 | — |
| **lm** | | | | |
| | MLE.score † | — | 0.23 | — |
| | Lidstone.score † | — | 0.20 | — |
| | Laplace.score † | — | 0.20 | — |
| | StupidBackoff.score † | — | 0.16 | — |
| | KneserNeyInterpolated.score † | — | 0.18 | — |
| | WittenBellInterpolated.score † | — | 0.18 | — |
| **ccg** | | | | |
| | CCG from_string | 0.81 | 0.37 | **2.2×** |
| **chunk** | | | | |
| | RegexpParser.parse | 2.01 | 0.25 | **7.9×** |
| **cluster** | | | | |
| | KMeansClusterer.cluster | 1.64 | 0.26 | **6.4×** |
| **parse** | | | | |
| | EarleyChartParser.parse | 7.37 | 0.51 | **14.4×** |
| | CFG.from_string | 0.05 | 0.002 | **24.1×** |
| **translate** | | | | |
| | bleu | 0.03 | 0.004 | **9.0×** |
| **chat** | | | | |
| | Chat.respond | 0.001 | 0.0003 | **3.0×** |
| **tree** | | | | |
| | Tree.from_string | 3.34 | 0.33 | **10.1×** |
| **sem** | | | | |
| | Expression.fromstring | 17.45 | 0.60 | **29.0×** |
| **inference** | | | | |
| | TableauProver.prove † | — | 0.0005 | — |
| | ResolutionProver.prove † | — | 0.0007 | — |
| | DiscourseThread.answer_question † | — | 0.0019 | — |
| | DefaultReasoner.extensions † | — | 4.54 | — |

† fastNLTK-only — no NLTK comparison available.

---

## Module Leaderboard

| Module | Geo Mean Speedup | Best Single | Key Engine |
|--------|-----------------|-------------|------------|
| metrics | **140.0×** | 180× (edit_distance) | Pure algorithmic port, zero Python overhead |
| tokenize | **13.3×** | 655× (TextTiling) | SIMD memchr3 + char scanner + byte-level segmentation |
| parse | **18.6×** | 24× (CFG) | Earley + CFG parsing |
| sem | **29.0×** | 29× (Expression) | FOL expression parser |
| collocations | **12.7×** | 18× (Quadgram) | FastMap ngram frequency counting |
| tree | **10.1×** | 10× | Tree bracket parser |
| translate | **9.0×** | 9× (BLEU) | BLEU in Rust |
| stem | **8.5×** | 17× (Lancaster) | 124-rule NLTK port |
| classify | **7.8×** | 495× (Maxent) | GIS training, fully optimized inner loop |
| chunk | **7.9×** | 8× | Regexp chunk parser |
| cluster | **6.4×** | 6× | K-means in Rust |
| tag | **3.3×** | 8× (RegexpTagger) | u64 feature IDs, integer Viterbi |
| probability | **2.9×** | 4× (FreqDist) | FreqDist/ConditionalFreqDist |
| ccg | **2.2×** | 2× | CCG category parsing |
| chat | **3.0×** | 3× | Eliza chatbot |
