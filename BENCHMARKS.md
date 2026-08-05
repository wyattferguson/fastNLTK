# Benchmarks

> **Last updated:** 2026-08-05 (v0.5.5, release build)
> **Geometric mean: 9.0× vs NLTK** across 51 compared benchmarks (68 total).

> Run benchmarks: `python -m benchmarks.run --save`
> Fixtures: NLTK Gutenberg corpus (~200KB medium, ~5KB tiny).

> [!NOTE]
> HMM tagger optimized in v0.5.3: integer tag IDs + flat matrices —
> eliminates `String::clone()` in the O(N × T²) Viterbi inner loop (85× speedup).
> ConditionalFreqDist now shares `FreqDist` references so mutations via
> `cfd[cond][sample] = value` propagate correctly.
> ISRI and RSLP stemmers delegate to NLTK in the Python wrapper (`fastnltk.stem`)
> for byte-identical output. The raw Rust `_rust` versions are benchmarked below
> but the user-facing interface matches NLTK exactly.

---

## Highlights

| Operation | NLTK (ms) | fastNLTK (ms) | Speedup | Notes |
|---|---|---|---|---|
| **TextTiling** | 22548.09 | **33.45** | **674×** | SIMD memchr3 + byte-level segmentation |
| **Maxent.train** | 33.87 | **0.09** | **373×** | GIS training, fully optimized inner loop |
| **windowdiff** | 2.38 | **0.01** | **174×** | Pure algorithmic port, zero Python overhead |
| **edit_distance** | 2.46 | **0.01** | **248×** | Damerau-Levenshtein in Rust |
| **HiddenMarkovModelTagger** | 8.75 | **0.10** | **85×** | Integer Viterbi, zero-alloc inner loop |
| **pk** | 2.22 | **0.03** | **83×** | Segmentation metric in Rust |
| **TreebankWordDetokenizer.detokenize** | 6.87 | **0.14** | **51×** | Single-pass undo |
| **SentimentIntensityAnalyzer.polarity_scores** | 68.43 | **1.80** | **38×** | PHF lexicon, exact NLTK scoring |
| **PunktSentence** | 14.47 | **0.41** | **35×** | Byte-level sentence scan |
| **Expression.fromstring** | 17.00 | **0.55** | **31×** | FOL parser in Rust |
| **SExpr** | 0.36 | **0.01** | **30×** | S-expression parser |
| **Tweet** | 84.88 | **4.34** | **20×** | LazyLock regexes |
| **CFG.from_string** | 0.05 | **0.00** | **22×** | Grammar parser in Rust |
| **LancasterStemmer.stem** | 33.61 | **1.80** | **19×** | Full 124-rule NLTK port |
| **QuadgramCollocationFinder.from_words** | 100.09 | **5.79** | **17×** | FastMap ngram counting |

---

## Full Results (68 benchmarks)

Benchmarks grouped by module. Numbers from `python -m benchmarks.run --save` on release build.

| Module | Benchmark | NLTK (ms) | fastNLTK (ms) | Speedup |
|--------|-----------|-----------|---------------|---------|
| **tokenize** | | | | |
| | ToktokTokenizer.tokenize | 18.70 | 4.46 | **4.2×** |
| | MWETokenizer.tokenize | 1.16 | 1.00 | 1.2× |
| | RegexpTokenizer.tokenize | 4.51 | 4.07 | 1.1× |
| | SpaceTokenizer.tokenize | 1.11 | 1.53 | 0.7× |
| | TreebankWordTokenizer.tokenize | 42.74 | 4.42 | **9.7×** |
| | TweetTokenizer.tokenize | 84.88 | 4.34 | **19.6×** |
| | TextTilingTokenizer.tokenize | 22548.09 | 33.45 | **674.1×** |
| | SExprTokenizer.tokenize | 0.36 | 0.01 | **30.4×** |
| | PunktSentenceTokenizer.tokenize | 14.47 | 0.41 | **35.2×** |
| | TabTokenizer.tokenize | 0.08 | 0.02 | **3.6×** |
| | LineTokenizer.tokenize | 0.24 | 0.18 | 1.3× |
| | WhitespaceTokenizer.tokenize | 4.23 | 1.38 | **3.1×** |
| | WordPunctTokenizer.tokenize | 5.64 | 1.64 | **3.4×** |
| | BlanklineTokenizer.tokenize | 2.22 | 0.11 | **19.4×** |
| | logos_word_tokenize † | — | 1.61 | — |
| **stem** | | | | |
| | SnowballStemmer.stem | 22.28 | 1.86 | **12.0×** |
| | PorterStemmer.stem | 44.46 | 7.79 | **5.7×** |
| | LancasterStemmer.stem | 33.61 | 1.80 | **18.6×** |
| | WordNetLemmatizer.lemmatize | 6.38 | 0.78 | **8.2×** |
| | ARLSTem.stem | 1.67 | 0.44 | **3.7×** |
| | ISRIStemmer.stem | 2.03 | 0.22 | **9.2×** |
| | RSLPStemmer.stem † | — | 0.12 | — |
| | RegexpStemmer.stem † | — | 0.38 | — |
| **tag** | | | | |
| | PerceptronTagger.tag | 16.45 | 2.43 | **6.8×** |
| | HiddenMarkovModelTagger.tag | 8.75 | 0.10 | **85.0×** |
| | TnT.tag | 1.02 | 0.21 | **4.9×** |
| | DefaultTagger.tag | 1.26 | 1.15 | 1.1× |
| | UnigramTagger.tag | 1.65 | 1.01 | **1.6×** |
| | BigramTagger.tag | 3.10 | 1.01 | **3.1×** |
| | TrigramTagger.tag | 3.15 | 1.04 | **3.0×** |
| | RegexpTagger.tag | 9.77 | 1.13 | **8.7×** |
| | AffixTagger.tag | 2.35 | 1.24 | **1.9×** |
| **classify** | | | | |
| | NaiveBayesClassifier.train | 6.53 | 1.58 | **4.1×** |
| | NaiveBayesClassifier.classify | 0.01 | 0.00 | **8.8×** |
| | MaxentClassifier.train | 33.87 | 0.09 | **372.6×** |
| | TextCat.guess_language † | — | 4.31 | — |
| **probability** | | | | |
| | FreqDist.update | 20.59 | 5.46 | **3.8×** |
| | ConditionalFreqDist.inc | 5.20 | 1.92 | **2.7×** |
| | LaplaceProbDist.prob † | — | 0.0003 | — |
| | MLEProbDist.prob † | — | 0.0003 | — |
| **collocations** | | | | |
| | BigramCollocationFinder.from_words | 63.81 | 7.58 | **8.4×** |
| | TrigramCollocationFinder.from_words | 59.26 | 4.39 | **13.5×** |
| | QuadgramCollocationFinder.from_words | 100.09 | 5.79 | **17.3×** |
| **sentiment** | | | | |
| | SentimentIntensityAnalyzer.polarity_scores | 68.43 | 1.80 | **38.0×** |
| **metrics** | | | | |
| | windowdiff | 2.38 | 0.01 | **173.9×** |
| | pk | 2.22 | 0.03 | **82.8×** |
| | edit_distance | 2.46 | 0.01 | **248.1×** |
| | BigramAssocMeasures † | — | 0.0003 | — |
| **lm** | | | | |
| | MLE.score † | — | 0.16 | — |
| | Lidstone.score † | — | 0.14 | — |
| | Laplace.score † | — | 0.14 | — |
| | StupidBackoff.score † | — | 0.12 | — |
| | KneserNeyInterpolated.score † | — | 0.14 | — |
| | WittenBellInterpolated.score † | — | 0.15 | — |
| **ccg** | | | | |
| | CCG from_string | 0.77 | 0.30 | **2.6×** |
| **chunk** | | | | |
| | RegexpParser.parse | 1.59 | 0.20 | **7.8×** |
| **cluster** | | | | |
| | KMeansClusterer.cluster | 1.57 | 0.26 | **6.0×** |
| **parse** | | | | |
| | EarleyChartParser.parse | 6.90 | 9.59 | 0.7× |
| | CFG.from_string | 0.05 | 0.00 | **22.0×** |
| **translate** | | | | |
| | bleu | 0.03 | 0.00 | **8.1×** |
| **chat** | | | | |
| | Chat.respond | 0.00 | 0.00 | **3.0×** |
| **tree** | | | | |
| | Tree.from_string | 3.20 | 0.32 | **10.0×** |
| **sem** | | | | |
| | Expression.fromstring | 17.00 | 0.55 | **30.8×** |
| **inference** | | | | |
| | TableauProver.prove † | — | 0.0005 | — |
| | ResolutionProver.prove † | — | 0.0007 | — |
| | DiscourseThread.answer_question † | — | 0.0020 | — |
| | DefaultReasoner.extensions † | — | 5.03 | — |

† fastNLTK-only — no NLTK comparison available.

---

## Module Leaderboard

| Module | Geo Mean Speedup | Best Single | Key Engine |
|--------|-----------------|-------------|------------|
| metrics | **153×** | 248× (edit_distance) | Pure algorithmic port, zero Python overhead |
| sentiment | **38×** | 38× (SentimentIntensityAnalyzer) | PHF lexicon, exact NLTK algorithm |
| sem | **31×** | 31× (Expression) | FOL expression parser |
| classify | **24×** | 373× (Maxent.train) | GIS training, fully optimized inner loop |
| collocations | **13×** | 17× (QuadgramCollocationFinder) | FastMap ngram frequency counting |
| tree | **10×** | 10× (Tree.from_string) | Tree bracket parser |
| stem | **8×** | 19× (LancasterStemmer) | 124-rule NLTK port |
| translate | **8×** | 8× (bleu) | BLEU in Rust |
| chunk | **8×** | 8× (RegexpParser.parse) | Regexp chunk parser |
| tokenize | **7×** | 674× (TextTiling) | SIMD memchr3 + char scanner + byte-level segmentation |
| cluster | **6×** | 6× (KMeansClusterer.cluster) | K-means in Rust |
| tag | **5×** | 85× (HiddenMarkovModelTagger) | u64 feature IDs, integer Viterbi |
| parse | **4×** | 22× (CFG.from_string) | Earley + CFG parsing |
| probability | **3×** | 4× (FreqDist.update) | SmolStr-optimized FreqDist |
| chat | **3×** | 3× (Chat.respond) | Eliza chatbot |
| ccg | **3×** | 3× (CCG from_string) | CCG category parsing |
