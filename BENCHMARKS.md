# Benchmarks

> **Last updated:** 2025-07-16 (v0.4.1, Rust 1.97.1, release build)
> **Geometric mean: 8.9× vs NLTK** across 49 compared benchmarks (66 total).
>
> Run benchmarks: `python -m benchmarks.run --save`
> Fixtures: NLTK Gutenberg corpus (~200KB medium, ~5KB tiny).

---

## Highlights

| Operation | NLTK (ms) | fastNLTK (ms) | Speedup | Notes |
|---|---|---|---|---|
| **TextTilingTokenizer** | 22642.32 | **32.48** | **697×** | Sentence segmentation, SIMD-accelerated |
| **edit_distance** | 2.49 | **0.01** | **213×** | Damerau-Levenshtein in Rust |
| **windowdiff** | 2.38 | **0.01** | **162×** | Pure algorithmic port, zero Python overhead |
| **pk** | 2.19 | **0.02** | **98×** | Segmentation metric in Rust |
| **MaxentClassifier.train** | 33.54 | **0.09** | **367×** | GIS training, fully optimized inner loop |
| **SentimentIntensityAnalyzer** | 68.44 | **1.70** | **40×** | PHF lexicon, single-pass word scan |
| **SExprTokenizer** | 0.35 | **0.01** | **31×** | S-expression splitter |
| **PunktSentenceTokenizer** | 14.99 | **0.44** | **34×** | Byte-level sentence scan |
| **TreebankWordDetokenizer** | 6.99 | **0.22** | **32×** | Single-pass undo |
| **Expression.fromstring** | 17.13 | **0.57** | **30×** | FOL parser in Rust |
| **TweetTokenizer** | 84.24 | **3.30** | **26×** | LazyLock regexes |
| **CFG.from_string** | 0.05 | **0.002** | **23×** | Grammar parser in Rust |
| **QuadgramCollocationFinder** | 104.26 | **5.71** | **18×** | FastMap ngram counting |
| **LancasterStemmer** | 33.01 | **1.86** | **18×** | Full 124-rule NLTK port |
| **TrigramCollocationFinder** | 62.79 | **4.45** | **14×** | FastMap ngram counting |
| **EarleyChartParser.parse** | 6.95 | **0.50** | **14×** | Chart parser in Rust |
| **SnowballStemmer** | 22.08 | **1.97** | **11×** | rust-stemmers crate |
| **Tree.from_string** | 3.36 | **0.33** | **10×** | Bracket parser in Rust |
| **TreebankWordTokenizer** | 42.22 | **4.17** | **10×** | Single-pass char scanner |

---

## Full Results (66 benchmarks)

Benchmarks grouped by module. Numbers from `python -m benchmarks.run --save` on release build, Rust 1.97.1.

| Module | Benchmark | NLTK (ms) | fastNLTK (ms) | Speedup |
|--------|-----------|-----------|---------------|---------|
| **tokenize** | | | | |
| | ToktokTokenizer.tokenize | 18.70 | 4.28 | **4.4×** |
| | MWETokenizer.tokenize | 1.13 | 1.19 | 1.0× |
| | RegexpTokenizer.tokenize | 4.40 | 3.82 | 1.2× |
| | SpaceTokenizer.tokenize | 0.96 | 1.56 | 0.6× |
| | TreebankWordTokenizer.tokenize | 42.22 | 4.17 | **10.1×** |
| | TweetTokenizer.tokenize | 84.24 | 3.30 | **25.5×** |
| | TextTilingTokenizer.tokenize | 22642.32 | 32.48 | **697.0×** |
| | SExprTokenizer.tokenize | 0.35 | 0.01 | **30.8×** |
| | PunktSentenceTokenizer.tokenize | 14.99 | 0.44 | **33.7×** |
| | TreebankWordDetokenizer.detokenize | 6.99 | 0.22 | **32.4×** |
| | TabTokenizer.tokenize | 0.08 | 0.03 | **2.9×** |
| | LineTokenizer.tokenize | 0.25 | 0.18 | 1.4× |
| | WhitespaceTokenizer.tokenize | 4.36 | 1.24 | **3.5×** |
| | WordPunctTokenizer.tokenize | 5.89 | 1.55 | **3.8×** |
| | BlanklineTokenizer.tokenize | 2.21 | 0.10 | **21.1×** |
| | logos_word_tokenize † | — | 1.55 | — |
| **stem** | | | | |
| | SnowballStemmer.stem | 22.08 | 1.97 | **11.2×** |
| | PorterStemmer.stem | 44.88 | 6.60 | **6.8×** |
| | LancasterStemmer.stem | 33.01 | 1.86 | **17.8×** |
| | WordNetLemmatizer.lemmatize | 6.34 | 0.98 | **6.5×** |
| | ARLSTem.stem | 1.70 | 0.47 | **3.7×** |
| | ISRIStemmer.stem | 2.02 | 0.27 | **7.4×** |
| | RSLPStemmer.stem † | — | 0.18 | — |
| | RegexpStemmer.stem † | — | 0.43 | — |
| **tag** | | | | |
| | PerceptronTagger.tag | 17.45 | 2.56 | **6.8×** |
| | TnT.tag | 1.00 | 0.21 | **4.7×** |
| | DefaultTagger.tag | 1.32 | 1.22 | 1.1× |
| | UnigramTagger.tag | 1.71 | 1.06 | **1.6×** |
| | BigramTagger.tag | 2.77 | 1.06 | **2.6×** |
| | TrigramTagger.tag | 2.89 | 1.11 | **2.6×** |
| | RegexpTagger.tag | 9.95 | 1.28 | **7.8×** |
| **classify** | | | | |
| | NaiveBayesClassifier.train | 6.34 | 2.08 | **3.1×** |
| | NaiveBayesClassifier.classify | 0.01 | 0.00 | **6.6×** |
| | MaxentClassifier.train | 33.54 | 0.09 | **366.9×** |
| | TextCat.guess_language † | — | 4.37 | — |
| **probability** | | | | |
| | FreqDist.update | 20.67 | 6.11 | **3.4×** |
| | ConditionalFreqDist.inc | 5.06 | 2.92 | **1.7×** |
| | LaplaceProbDist.prob † | — | 0.0004 | — |
| | MLEProbDist.prob † | — | 0.0004 | — |
| **collocations** | | | | |
| | BigramCollocationFinder.from_words | 63.11 | 7.23 | **8.7×** |
| | TrigramCollocationFinder.from_words | 62.79 | 4.45 | **14.1×** |
| | QuadgramCollocationFinder.from_words | 104.26 | 5.71 | **18.2×** |
| **sentiment** | | | | |
| | SentimentIntensityAnalyzer.polarity_scores | 68.44 | 1.70 | **40.2×** |
| **metrics** | | | | |
| | windowdiff | 2.38 | 0.01 | **161.7×** |
| | pk | 2.19 | 0.02 | **98.3×** |
| | edit_distance | 2.49 | 0.01 | **213.2×** |
| | BigramAssocMeasures † | — | 0.0003 | — |
| **lm** | | | | |
| | MLE.score † | — | 0.23 | — |
| | Lidstone.score † | — | 0.20 | — |
| | Laplace.score † | — | 0.21 | — |
| | StupidBackoff.score † | — | 0.16 | — |
| | KneserNeyInterpolated.score † | — | 0.18 | — |
| | WittenBellInterpolated.score † | — | 0.19 | — |
| **ccg** | | | | |
| | CCG from_string | 0.77 | 0.36 | **2.2×** |
| **chunk** | | | | |
| | RegexpParser.parse | 1.83 | 0.25 | **7.3×** |
| **cluster** | | | | |
| | KMeansClusterer.cluster | 1.61 | 0.30 | **5.5×** |
| **parse** | | | | |
| | EarleyChartParser.parse | 6.95 | 0.50 | **13.8×** |
| | CFG.from_string | 0.05 | 0.002 | **23.3×** |
| **translate** | | | | |
| | bleu | 0.03 | 0.004 | **8.8×** |
| **chat** | | | | |
| | Chat.respond | 0.001 | 0.0003 | **3.0×** |
| **tree** | | | | |
| | Tree.from_string | 3.36 | 0.33 | **10.1×** |
| **sem** | | | | |
| | Expression.fromstring | 17.13 | 0.57 | **29.9×** |
| **inference** | | | | |
| | TableauProver.prove † | — | 0.0005 | — |
| | ResolutionProver.prove † | — | 0.0007 | — |
| | DiscourseThread.answer_question † | — | 0.0019 | — |
| | DefaultReasoner.extensions † | — | 4.39 | — |

† fastNLTK-only — no NLTK comparison available.

---

## Module Leaderboard

| Module | Geo Mean Speedup | Best Single | Key Engine |
|--------|-----------------|-------------|------------|
| metrics | **150.0×** | 213× (edit_distance) | Pure algorithmic port, zero Python overhead |
| tokenize | **13.6×** | 697× (TextTiling) | SIMD memchr3 + char scanner + byte-level segmentation |
| parse | **17.9×** | 23× (CFG) | Earley + CFG parsing |
| sem | **29.9×** | 30× (Expression) | FOL expression parser |
| collocations | **13.0×** | 18× (Quadgram) | FastMap ngram frequency counting |
| sentiment | **40.2×** | 40× (VADER) | PHF lexicon, single-pass word scan |
| tree | **10.1×** | 10× | Tree bracket parser |
| translate | **8.8×** | 9× (BLEU) | BLEU in Rust |
| stem | **7.2×** | 18× (Lancaster) | 124-rule NLTK port |
| classify | **6.5×** | 367× (Maxent) | GIS training, fully optimized inner loop |
| chunk | **7.3×** | 7× | Regexp chunk parser |
| cluster | **5.5×** | 6× | K-means in Rust |
| tag | **3.2×** | 8× (RegexpTagger) | u64 feature IDs, integer Viterbi |
| probability | **2.4×** | 3× (FreqDist) | SmolStr-optimized FreqDist |
| ccg | **2.2×** | 2× | CCG category parsing |
| chat | **3.0×** | 3× | Eliza chatbot |
