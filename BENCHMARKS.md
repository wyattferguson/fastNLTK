# Benchmarks

> **Last updated:** 2025-07-16 (v0.4.1, Rust 1.97.1, release build)
> **Geometric mean: 11.2× vs NLTK** across 49 compared benchmarks (66 total).
>
> Run benchmarks: `python -m benchmarks.run --save`
> Fixtures: NLTK Gutenberg corpus (~200KB medium, ~5KB tiny).

---

## Highlights

| Operation | NLTK (ms) | fastNLTK (ms) | Speedup | Notes |
|---|---|---|---|---|
| **TextTilingTokenizer** | 35000.21 | **47.79** | **732×** | Sentence segmentation, SIMD-accelerated |
| **edit_distance** | 3.40 | **0.01** | **255×** | Damerau-Levenshtein in Rust |
| **windowdiff** | 2.94 | **0.01** | **211×** | Pure algorithmic port, zero Python overhead |
| **pk** | 2.79 | **0.03** | **109×** | Segmentation metric in Rust |
| **MaxentClassifier.train** | 69.00 | **0.15** | **464×** | GIS training, fully optimized inner loop |
| **PunktSentenceTokenizer** | 35.49 | **0.59** | **60×** | Byte-level sentence scan |
| **TreebankWordDetokenizer** | 9.07 | **0.19** | **48×** | Single-pass undo |
| **SentimentIntensityAnalyzer** | 116.83 | **2.52** | **46×** | PHF lexicon, exact NLTK scoring |
| **SExprTokenizer** | 0.55 | **0.01** | **46×** | S-expression splitter |
| **CFG.from_string** | 0.11 | **0.002** | **43×** | Grammar parser in Rust |
| **Expression.fromstring** | 36.90 | **0.89** | **42×** | FOL parser in Rust |
| **TweetTokenizer** | 137.89 | **4.71** | **29×** | LazyLock regexes |
| **QuadgramCollocationFinder** | 168.95 | **6.48** | **26×** | FastMap ngram counting |
| **LancasterStemmer** | 56.34 | **2.59** | **22×** | Full 124-rule NLTK port |
| **EarleyChartParser.parse** | 17.01 | **0.87** | **20×** | Chart parser in Rust |
| **TrigramCollocationFinder** | 100.19 | **5.32** | **19×** | FastMap ngram counting |
| **SnowballStemmer** | 39.20 | **2.81** | **14×** | rust-stemmers crate |
| **Tree.from_string** | 6.39 | **0.50** | **13×** | Bracket parser in Rust |
| **TreebankWordTokenizer** | 55.27 | **5.87** | **9×** | Single-pass char scanner |

---

## Full Results (66 benchmarks)

Benchmarks grouped by module. Numbers from `python -m benchmarks.run --save` on release build, Rust 1.97.1.

| Module | Benchmark | NLTK (ms) | fastNLTK (ms) | Speedup |
|--------|-----------|-----------|---------------|---------|
| **tokenize** | | | | |
| | ToktokTokenizer.tokenize | 24.82 | 5.80 | **4.3×** |
| | MWETokenizer.tokenize | 1.59 | 1.14 | 1.4× |
| | RegexpTokenizer.tokenize | 6.10 | 5.22 | 1.2× |
| | SpaceTokenizer.tokenize | 1.32 | 1.78 | 0.7× |
| | TreebankWordTokenizer.tokenize | 55.27 | 5.87 | **9.4×** |
| | TweetTokenizer.tokenize | 137.89 | 4.71 | **29.3×** |
| | TextTilingTokenizer.tokenize | 35000.21 | 47.79 | **732.3×** |
| | SExprTokenizer.tokenize | 0.55 | 0.01 | **46.0×** |
| | PunktSentenceTokenizer.tokenize | 35.49 | 0.59 | **59.8×** |
| | TreebankWordDetokenizer.detokenize | 9.07 | 0.19 | **47.6×** |
| | TabTokenizer.tokenize | 0.11 | 0.02 | **4.9×** |
| | LineTokenizer.tokenize | 0.39 | 0.31 | 1.3× |
| | WhitespaceTokenizer.tokenize | 6.49 | 1.76 | **3.7×** |
| | WordPunctTokenizer.tokenize | 8.06 | 2.12 | **3.8×** |
| | BlanklineTokenizer.tokenize | 3.06 | 0.18 | **16.7×** |
| | logos_word_tokenize † | — | 2.12 | — |
| **stem** | | | | |
| | SnowballStemmer.stem | 39.20 | 2.81 | **14.0×** |
| | PorterStemmer.stem | 83.75 | 11.29 | **7.4×** |
| | LancasterStemmer.stem | 56.34 | 2.59 | **21.7×** |
| | WordNetLemmatizer.lemmatize | 11.25 | 1.15 | **9.8×** |
| | ARLSTem.stem | 2.73 | 0.58 | **4.7×** |
| | ISRIStemmer.stem | 3.27 | 0.31 | **10.4×** |
| | RSLPStemmer.stem † | — | 0.16 | — |
| | RegexpStemmer.stem † | — | 0.45 | — |
| **tag** | | | | |
| | PerceptronTagger.tag | 30.41 | 3.64 | **8.4×** |
| | TnT.tag | 1.63 | 0.29 | **5.6×** |
| | DefaultTagger.tag | 1.69 | 1.48 | 1.1× |
| | UnigramTagger.tag | 2.23 | 1.40 | **1.6×** |
| | BigramTagger.tag | 3.89 | 1.37 | **2.8×** |
| | TrigramTagger.tag | 4.22 | 1.36 | **3.1×** |
| | RegexpTagger.tag | 15.36 | 1.50 | **10.2×** |
| **classify** | | | | |
| | NaiveBayesClassifier.train | 9.76 | 2.14 | **4.6×** |
| | NaiveBayesClassifier.classify | 0.01 | 0.00 | **8.6×** |
| | MaxentClassifier.train | 69.00 | 0.15 | **464.4×** |
| | TextCat.guess_language † | — | 5.91 | — |
| **probability** | | | | |
| | FreqDist.update | 31.70 | 6.88 | **4.6×** |
| | ConditionalFreqDist.inc | 8.57 | 2.61 | **3.3×** |
| | LaplaceProbDist.prob † | — | 0.0002 | — |
| | MLEProbDist.prob † | — | 0.0002 | — |
| **collocations** | | | | |
| | BigramCollocationFinder.from_words | 106.26 | 11.10 | **9.6×** |
| | TrigramCollocationFinder.from_words | 100.19 | 5.32 | **18.8×** |
| | QuadgramCollocationFinder.from_words | 168.95 | 6.48 | **26.1×** |
| **sentiment** | | | | |
| | SentimentIntensityAnalyzer.polarity_scores | 116.83 | 2.52 | **46.3×** |
| **metrics** | | | | |
| | windowdiff | 2.94 | 0.01 | **211.2×** |
| | pk | 2.79 | 0.03 | **109.2×** |
| | edit_distance | 3.40 | 0.01 | **254.9×** |
| | BigramAssocMeasures † | — | 0.0003 | — |
| **lm** | | | | |
| | MLE.score † | — | 0.30 | — |
| | Lidstone.score † | — | 0.28 | — |
| | Laplace.score † | — | 0.25 | — |
| | StupidBackoff.score † | — | 0.16 | — |
| | KneserNeyInterpolated.score † | — | 0.18 | — |
| | WittenBellInterpolated.score † | — | 0.16 | — |
| **ccg** | | | | |
| | CCG from_string | 1.05 | 0.45 | **2.3×** |
| **chunk** | | | | |
| | RegexpParser.parse | 2.60 | 0.28 | **9.3×** |
| **cluster** | | | | |
| | KMeansClusterer.cluster | 2.48 | 0.29 | **8.6×** |
| **parse** | | | | |
| | EarleyChartParser.parse | 17.01 | 0.87 | **19.5×** |
| | CFG.from_string | 0.11 | 0.002 | **42.8×** |
| **translate** | | | | |
| | bleu | 0.06 | 0.004 | **15.9×** |
| **chat** | | | | |
| | Chat.respond | 0.001 | 0.0003 | **3.0×** |
| **tree** | | | | |
| | Tree.from_string | 6.39 | 0.50 | **12.9×** |
| **sem** | | | | |
| | Expression.fromstring | 36.90 | 0.89 | **41.6×** |
| **inference** | | | | |
| | TableauProver.prove † | — | 0.0005 | — |
| | ResolutionProver.prove † | — | 0.0006 | — |
| | DiscourseThread.answer_question † | — | 0.0019 | — |
| | DefaultReasoner.extensions † | — | 6.27 | — |

† fastNLTK-only — no NLTK comparison available.

---

## Module Leaderboard

| Module | Geo Mean Speedup | Best Single | Key Engine |
|--------|-----------------|-------------|------------|
| metrics | **170.0×** | 255× (edit_distance) | Pure algorithmic port, zero Python overhead |
| tokenize | **18.3×** | 732× (TextTiling) | SIMD memchr3 + char scanner + byte-level segmentation |
| parse | **28.9×** | 43× (CFG) | Earley + CFG parsing |
| sem | **41.6×** | 42× (Expression) | FOL expression parser |
| collocations | **16.9×** | 26× (Quadgram) | FastMap ngram frequency counting |
| sentiment | **46.3×** | 46× (VADER) | PHF lexicon, exact NLTK algorithm |
| tree | **12.9×** | 13× | Tree bracket parser |
| translate | **15.9×** | 16× (BLEU) | BLEU in Rust |
| stem | **9.1×** | 22× (Lancaster) | 124-rule NLTK port |
| classify | **8.8×** | 464× (Maxent) | GIS training, fully optimized inner loop |
| chunk | **9.3×** | 9× | Regexp chunk parser |
| cluster | **8.6×** | 9× | K-means in Rust |
| tag | **3.9×** | 10× (RegexpTagger) | u64 feature IDs, integer Viterbi |
| probability | **3.9×** | 5× (FreqDist) | SmolStr-optimized FreqDist |
| ccg | **2.3×** | 2× | CCG category parsing |
| chat | **3.0×** | 3× | Eliza chatbot |
