# Benchmarks

> **Last updated:** 2026-07-22 (v0.5.4, release build)
> **Geometric mean: 9.5× vs NLTK** across 51 compared benchmarks (68 total).
>
> Run benchmarks: `python -m benchmarks.run --save`
> Fixtures: NLTK Gutenberg corpus (~200KB medium, ~5KB tiny).

> [!NOTE]
> HMM tagger optimized in v0.5.3: integer tag IDs + flat matrices —
> eliminates `String::clone()` in the O(N × T²) Viterbi inner loop (88× speedup).
> ConditionalFreqDist now shares `FreqDist` references so mutations via
> `cfd[cond][sample] = value` propagate correctly.
> ISRI and RSLP stemmers delegate to NLTK in the Python wrapper (`fastnltk.stem`)
> for byte-identical output. The raw Rust `_rust` versions are benchmarked below
> but the user-facing interface matches NLTK exactly.

---

## Highlights

| Operation | NLTK (ms) | fastNLTK (ms) | Speedup | Notes |
|---|---|---|---|---|
| **TextTilingTokenizer** | 22043.28 | **31.58** | **698×** | SIMD memchr3 + byte-level segmentation |
| **MaxentClassifier.train** | 32.73 | **0.08** | **431×** | GIS training, fully optimized inner loop |
| **windowdiff** | 2.38 | **0.01** | **174×** | Pure algorithmic port, zero Python overhead |
| **edit_distance** | 2.44 | **0.02** | **144×** | Damerau-Levenshtein in Rust |
| **HiddenMarkovModelTagger** | 8.58 | **0.10** | **88×** | Integer Viterbi, zero-alloc inner loop |
| **pk** | 2.23 | **0.03** | **83×** | Segmentation metric in Rust |
| **TreebankWordDetokenizer** | 6.69 | **0.12** | **54×** | Single-pass undo |
| **SentimentIntensityAnalyzer** | 67.54 | **1.79** | **38×** | PHF lexicon, exact NLTK scoring |
| **PunktSentenceTokenizer** | 14.28 | **0.43** | **33×** | Byte-level sentence scan |
| **Expression.fromstring** | 16.06 | **0.54** | **30×** | FOL parser in Rust |
| **SExprTokenizer** | 0.35 | **0.01** | **30×** | S-expression parser |
| **TweetTokenizer** | 83.00 | **3.26** | **25×** | LazyLock regexes |
| **CFG.from_string** | 0.05 | **0.00** | **23×** | Grammar parser in Rust |
| **LancasterStemmer** | 31.50 | **1.42** | **22×** | Full 124-rule NLTK port |
| **QuadgramCollocationFinder** | 98.74 | **5.11** | **19×** | FastMap ngram counting |

---

## Full Results (66 benchmarks)

Benchmarks grouped by module. Numbers from `python -m benchmarks.run --save` on release build.

| Module | Benchmark | NLTK (ms) | fastNLTK (ms) | Speedup |
|--------|-----------|-----------|---------------|---------|
| **tokenize** | | | | |
| | ToktokTokenizer.tokenize | 18.23 | 4.26 | **4.3×** |
| | MWETokenizer.tokenize | 1.17 | 0.90 | 1.3× |
| | RegexpTokenizer.tokenize | 4.41 | 3.76 | 1.2× |
| | SpaceTokenizer.tokenize | 1.07 | 1.47 | 0.7× |
| | TreebankWordTokenizer.tokenize | 42.28 | 4.03 | **10.5×** |
| | TweetTokenizer.tokenize | 83.00 | 3.26 | **25.5×** |
| | TextTilingTokenizer.tokenize | 22043.28 | 31.58 | **698.0×** |
| | SExprTokenizer.tokenize | 0.35 | 0.01 | **29.9×** |
| | PunktSentenceTokenizer.tokenize | 14.28 | 0.43 | **33.3×** |
| | TreebankWordDetokenizer.detokenize | 6.69 | 0.12 | **54.0×** |
| | TabTokenizer.tokenize | 0.08 | 0.02 | **3.9×** |
| | LineTokenizer.tokenize | 0.24 | 0.18 | 1.3× |
| | WhitespaceTokenizer.tokenize | 4.19 | 1.24 | **3.4×** |
| | WordPunctTokenizer.tokenize | 5.47 | 1.52 | **3.6×** |
| | BlanklineTokenizer.tokenize | 2.20 | 0.10 | **22.8×** |
| | logos_word_tokenize † | — | 1.51 | — |
| **stem** | | | | |
| | SnowballStemmer.stem | 21.60 | 1.77 | **12.2×** |
| | PorterStemmer.stem | 42.95 | 6.77 | **6.3×** |
| | LancasterStemmer.stem | 31.50 | 1.42 | **22.2×** |
| | WordNetLemmatizer.lemmatize | 6.21 | 0.74 | **8.4×** |
| | ARLSTem.stem | 1.69 | 0.37 | **4.6×** |
| | ISRIStemmer.stem | 1.99 | 0.20 | **9.9×** |
| | RSLPStemmer.stem † | — | 0.11 | — |
| | RegexpStemmer.stem † | — | 0.31 | — |
| **tag** | | | | |
| | PerceptronTagger.tag | 16.30 | 2.43 | **6.7×** |
| | HiddenMarkovModelTagger.tag | 8.58 | 0.10 | **88.3×** |
| | TnT.tag | 1.09 | 0.20 | **5.4×** |
| | DefaultTagger.tag | 1.25 | 1.03 | 1.2× |
| | UnigramTagger.tag | 1.75 | 0.95 | **1.8×** |
| | BigramTagger.tag | 2.87 | 0.94 | **3.0×** |
| | TrigramTagger.tag | 3.02 | 0.97 | **3.1×** |
| | RegexpTagger.tag | 9.57 | 1.07 | **9.0×** |
| | AffixTagger.tag | 2.45 | 1.12 | **2.2×** |
| **classify** | | | | |
| | NaiveBayesClassifier.train | 6.14 | 1.59 | **3.9×** |
| | NaiveBayesClassifier.classify | 0.01 | 0.00 | **8.8×** |
| | MaxentClassifier.train | 32.73 | 0.08 | **430.7×** |
| | TextCat.guess_language † | — | 4.19 | — |
| **probability** | | | | |
| | FreqDist.update | 19.49 | 4.49 | **4.3×** |
| | ConditionalFreqDist.inc | 5.20 | 1.79 | **2.9×** |
| | LaplaceProbDist.prob † | — | 0.00 | — |
| | MLEProbDist.prob † | — | 0.00 | — |
| **collocations** | | | | |
| | BigramCollocationFinder.from_words | 62.17 | 6.43 | **9.7×** |
| | TrigramCollocationFinder.from_words | 57.39 | 3.77 | **15.2×** |
| | QuadgramCollocationFinder.from_words | 98.74 | 5.11 | **19.3×** |
| **sentiment** | | | | |
| | SentimentIntensityAnalyzer.polarity_scores | 67.54 | 1.79 | **37.6×** |
| **metrics** | | | | |
| | windowdiff | 2.38 | 0.01 | **173.6×** |
| | pk | 2.23 | 0.03 | **83.2×** |
| | edit_distance | 2.44 | 0.02 | **143.6×** |
| | BigramAssocMeasures † | — | 0.00 | — |
| **lm** | | | | |
| | MLE.score † | — | 0.16 | — |
| | Lidstone.score † | — | 0.15 | — |
| | Laplace.score † | — | 0.14 | — |
| | StupidBackoff.score † | — | 0.10 | — |
| | KneserNeyInterpolated.score † | — | 0.12 | — |
| | WittenBellInterpolated.score † | — | 0.12 | — |
| **ccg** | | | | |
| | CCG from_string | 0.81 | 0.27 | **3.0×** |
| **chunk** | | | | |
| | RegexpParser.parse | 1.57 | 0.18 | **8.5×** |
| **cluster** | | | | |
| | KMeansClusterer.cluster | 1.64 | 0.25 | **6.4×** |
| **parse** | | | | |
| | EarleyChartParser.parse | 6.35 | 9.21 | 0.7× |
| | CFG.from_string | 0.05 | 0.00 | **22.5×** |
| **translate** | | | | |
| | bleu | 0.03 | 0.00 | **9.6×** |
| **chat** | | | | |
| | Chat.respond | 0.001 | 0.0002 | **3.0×** |
| **tree** | | | | |
| | Tree.from_string | 3.21 | 0.30 | **10.6×** |
| **sem** | | | | |
| | Expression.fromstring | 16.06 | 0.54 | **29.8×** |
| **inference** | | | | |
| | TableauProver.prove † | — | 0.0004 | — |
| | ResolutionProver.prove † | — | 0.0006 | — |
| | DiscourseThread.answer_question † | — | 0.0017 | — |
| | DefaultReasoner.extensions † | — | 4.42 | — |

† fastNLTK-only — no NLTK comparison available.

---

## Module Leaderboard

| Module | Geo Mean Speedup | Best Single | Key Engine |
|--------|-----------------|-------------|------------|
| metrics | **128×** | 174× (windowdiff) | Pure algorithmic port, zero Python overhead |
| sentiment | **38×** | 38× (VADER) | PHF lexicon, exact NLTK algorithm |
| sem | **30×** | 30× (Expression) | FOL expression parser |
| classify | **25×** | 431× (Maxent) | GIS training, fully optimized inner loop |
| collocations | **14×** | 19× (Quadgram) | FastMap ngram frequency counting |
| tree | **11×** | 11× | Tree bracket parser |
| translate | **10×** | 10× (BLEU) | BLEU in Rust |
| stem | **9×** | 22× (Lancaster) | 124-rule NLTK port |
| chunk | **9×** | 9× | Regexp chunk parser |
| tokenize | **8×** | 698× (TextTiling) | SIMD memchr3 + char scanner + byte-level segmentation |
| cluster | **6×** | 6× | K-means in Rust |
| tag | **5×** | 88× (HMM) | u64 feature IDs, integer Viterbi |
| parse | **4×** | 23× (CFG) | Earley + CFG parsing |
| probability | **4×** | 4× (FreqDist) | SmolStr-optimized FreqDist |
| ccg | **3×** | 3× | CCG category parsing |
| chat | **3×** | 3× | Eliza chatbot |
