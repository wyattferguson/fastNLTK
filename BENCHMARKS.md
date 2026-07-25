# Benchmarks

> **Last updated:** 2026-07-25 (v0.5.4, release build)
> **Geometric mean: 10.2× vs NLTK** across 51 compared benchmarks (68 total).
>
> Run benchmarks: `python -m benchmarks.run --save`
> Fixtures: NLTK Gutenberg corpus (~200KB medium, ~5KB tiny).

> [!NOTE]
> v0.5.4 performance improvements: edit_distance now pre-collects chars (232× speedup, +40% from v0.5.3),
> HMM tagger uses flat 1D transition matrix for cache-friendly Viterbi (97×),
> Punkt abbreviates use single case-insensitive lookup,
> FreqDist.most_common() uses binary heap for top-k queries,
> NaiveBayes classifies via hashbrown::HashMap,
> regex cache uses RwLock for concurrent reads,
> chunk parser eliminates per-rule tag vector clone,
> Jaro similarity uses bool vec for O(1) matched-set lookups.

---

## Highlights

| Operation | NLTK (ms) | fastNLTK (ms) | Speedup | Notes |
|---|---|---|---|---|
| **MaxentClassifier.train** | 68.26 | **0.15** | **455×** | GIS training, fully optimized inner loop |
| **edit_distance** | 4.64 | **0.02** | **232×** | Damerau-Levenshtein in Rust, char pre-collection |
| **windowdiff** | 3.70 | **0.03** | **123×** | Pure algorithmic port, zero Python overhead |
| **PunktSentenceTokenizer** | 17.72 | **0.16** | **111×** | Byte-level sentence scan, single abbrev lookup |
| **HiddenMarkovModelTagger** | 17.45 | **0.18** | **98×** | Integer Viterbi, flat cache-friendly matrix |
| **TextTilingTokenizer** | 2.73 | **0.03** | **91×** | Sentence segmentation in Rust |
| **TreebankWordDetokenizer** | 11.14 | **0.20** | **56×** | Single-pass undo |
| **pk** | 3.59 | **0.07** | **51×** | Segmentation metric in Rust |
| **TweetTokenizer** | 71.45 | **1.54** | **47×** | LazyLock regexes |
| **Expression.fromstring** | 35.02 | **0.97** | **36×** | FOL parser in Rust |
| **BlanklineTokenizer** | 1.45 | **0.04** | **34×** | SIMD-accelerated scanning |
| **SExprTokenizer** | 0.66 | **0.02** | **33×** | S-expression tokenizer |
| **LancasterStemmer** | 51.70 | **1.80** | **29×** | Full 124-rule NLTK port |
| **CFG.from_string** | 0.11 | **0.00** | **28×** | Grammar parser in Rust |
| **VADER Sentiment** | 29.69 | **1.11** | **27×** | PHF lexicon, exact NLTK scoring |
| **QuadgramCollocationFinder** | 95.16 | **6.58** | **15×** | FastMap ngram counting |
| **SnowballStemmer** | 43.94 | **2.84** | **16×** | rust-stemmers crate |

---

## Full Results (68 benchmarks)

Benchmarks grouped by module. Numbers from `python -m benchmarks.run --save` on release build.

| Module | Benchmark | NLTK (ms) | fastNLTK (ms) | Speedup |
|--------|-----------|-----------|---------------|---------|
| **tokenize** | | | | |
| | ToktokTokenizer.tokenize | 11.14 | 1.84 | **6.1×** |
| | MWETokenizer.tokenize | 1.20 | 0.82 | 1.5× |
| | RegexpTokenizer.tokenize | 2.75 | 1.88 | 1.5× |
| | SpaceTokenizer.tokenize | 0.38 | 0.55 | 0.7× |
| | TreebankWordTokenizer.tokenize | 27.53 | 2.73 | **10.1×** |
| | TweetTokenizer.tokenize | 71.45 | 1.54 | **46.5×** |
| | TextTilingTokenizer.tokenize | 2.73 | 0.03 | **91.0×** |
| | SExprTokenizer.tokenize | 0.66 | 0.02 | **33.6×** |
| | PunktSentenceTokenizer.tokenize | 17.72 | 0.16 | **110.8×** |
| | TreebankWordDetokenizer.detokenize | 11.14 | 0.20 | **55.7×** |
| | TabTokenizer.tokenize | 0.04 | 0.02 | 1.8× |
| | LineTokenizer.tokenize | 0.09 | 0.01 | **6.9×** |
| | WhitespaceTokenizer.tokenize | 2.58 | 0.56 | **4.6×** |
| | WordPunctTokenizer.tokenize | 4.09 | 0.63 | **6.5×** |
| | BlanklineTokenizer.tokenize | 1.45 | 0.04 | **33.7×** |
| | logos_word_tokenize † | — | 0.60 | — |
| **stem** | | | | |
| | SnowballStemmer.stem | 43.94 | 2.84 | **15.5×** |
| | PorterStemmer.stem | 101.65 | 15.47 | **6.6×** |
| | LancasterStemmer.stem | 51.70 | 1.80 | **28.7×** |
| | WordNetLemmatizer.lemmatize | 12.16 | 1.14 | **10.7×** |
| | ARLSTem.stem | 2.92 | 0.75 | **3.9×** |
| | ISRIStemmer.stem | 3.93 | 0.35 | **11.1×** |
| | RSLPStemmer.stem † | — | 0.19 | — |
| | RegexpStemmer.stem † | — | 0.46 | — |
| **tag** | | | | |
| | PerceptronTagger.tag | 42.12 | 6.28 | **6.7×** |
| | HiddenMarkovModelTagger.tag | 17.45 | 0.18 | **97.6×** |
| | TnT.tag | 2.26 | 0.46 | **4.9×** |
| | DefaultTagger.tag | 2.09 | 1.97 | 1.1× |
| | UnigramTagger.tag | 3.26 | 1.59 | **2.1×** |
| | BigramTagger.tag | 5.03 | 1.57 | **3.2×** |
| | TrigramTagger.tag | 5.22 | 1.73 | **3.0×** |
| | RegexpTagger.tag | 20.82 | 1.98 | **10.5×** |
| | AffixTagger.tag | 4.17 | 1.91 | **2.2×** |
| **classify** | | | | |
| | NaiveBayesClassifier.train | 12.57 | 2.53 | **5.0×** |
| | NaiveBayesClassifier.classify | 0.01 | 0.001 | **11.4×** |
| | MaxentClassifier.train | 68.26 | 0.15 | **455.1×** |
| | TextCat.guess_language † | — | 8.83 | — |
| **probability** | | | | |
| | FreqDist.update | 26.46 | 4.42 | **6.0×** |
| | ConditionalFreqDist.inc | 5.91 | 1.90 | **3.1×** |
| | LaplaceProbDist.prob † | — | 0.0004 | — |
| | MLEProbDist.prob † | — | 0.0004 | — |
| **collocations** | | | | |
| | BigramCollocationFinder.from_words | 63.63 | 5.82 | **10.9×** |
| | TrigramCollocationFinder.from_words | 51.70 | 2.42 | **21.4×** |
| | QuadgramCollocationFinder.from_words | 95.16 | 6.58 | **14.5×** |
| **sentiment** | | | | |
| | SentimentIntensityAnalyzer.polarity_scores | 29.69 | 1.11 | **26.9×** |
| **metrics** | | | | |
| | windowdiff | 3.70 | 0.03 | **123.3×** |
| | pk | 3.59 | 0.07 | **51.3×** |
| | edit_distance | 4.64 | 0.02 | **240.6×** |
| | BigramAssocMeasures † | — | 0.0004 | — |
| **lm** | | | | |
| | MLE.score † | — | 0.27 | — |
| | Lidstone.score † | — | 0.23 | — |
| | Laplace.score † | — | 0.23 | — |
| | StupidBackoff.score † | — | 0.18 | — |
| | KneserNeyInterpolated.score † | — | 0.22 | — |
| | WittenBellInterpolated.score † | — | 0.22 | — |
| **ccg** | | | | |
| | CCG from_string | 1.61 | 0.59 | **2.7×** |
| **chunk** | | | | |
| | RegexpParser.parse | 2.42 | 0.28 | **8.6×** |
| **cluster** | | | | |
| | KMeansClusterer.cluster | 2.81 | 0.78 | **3.6×** |
| **parse** | | | | |
| | EarleyChartParser.parse | 15.52 | 21.46 | 0.7× |
| | CFG.from_string | 0.11 | 0.004 | **27.8×** |
| **translate** | | | | |
| | bleu | 0.06 | 0.006 | **10.7×** |
| **chat** | | | | |
| | Chat.respond | 0.002 | 0.0007 | **3.1×** |
| **tree** | | | | |
| | Tree.from_string | 6.60 | 0.96 | **6.9×** |
| **sem** | | | | |
| | Expression.fromstring | 35.02 | 0.97 | **36.2×** |
| **inference** | | | | |
| | TableauProver.prove † | — | 0.0013 | — |
| | ResolutionProver.prove † | — | 0.0016 | — |
| | DiscourseThread.answer_question † | — | 0.0047 | — |
| | DefaultReasoner.extensions † | — | 9.40 | — |

† fastNLTK-only — no NLTK comparison available.

---

## Module Leaderboard

| Module | Geo Mean Speedup | Best Single | Key Engine |
|--------|-----------------|-------------|------------|
| metrics | **114×** | 241× (edit_distance) | Pure algorithmic port, char pre-collection |
| sem | **36×** | 36× (Expression) | FOL expression parser |
| classify | **29×** | 455× (Maxent) | GIS training, fully optimized inner loop |
| sentiment | **27×** | 27× (VADER) | PHF lexicon, exact NLTK algorithm |
| collocations | **15×** | 21× (Trigram) | FastMap ngram frequency counting |
| stem | **11×** | 29× (Lancaster) | 124-rule NLTK port |
| tokenize | **10×** | 111× (Punkt) | SIMD memchr3 + char scanner |
| translate | **10×** | 10× (BLEU) | BLEU in Rust |
| chunk | **9×** | 9× (RegexpParser) | Regexp chunk parser, no tag clone |
| tree | **7×** | 7× (Tree) | Tree bracket parser |
| tag | **5×** | 98× (HMM) | u64 feature IDs, flat Viterbi matrix |
| probability | **4×** | 6× (FreqDist) | SmolStr + binary heap top-k |
| cluster | **4×** | 4× (KMeans) | K-means in Rust |
| parse | **4×** | 28× (CFG) | CFG grammar parser |
| ccg | **3×** | 3× (CCG) | CCG category parsing |
| chat | **3×** | 3× (Chat) | Eliza chatbot |

† EarleyChartParser excluded from tokenize geo mean (not yet optimized).
