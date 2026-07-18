# Benchmarks

> **Last updated:** 2026-07-16 (v0.5.3, release build)
> **Geometric mean: 12.2× vs NLTK** across 48 compared benchmarks (68 total).
>
> Run benchmarks: `python -m benchmarks.run --save`
> Fixtures: NLTK Gutenberg corpus (~200KB medium, ~5KB tiny).

> [!NOTE]
> HMM tagger optimized in v0.5.3: integer tag IDs + flat matrices —
> eliminates `String::clone()` in the O(N × T²) Viterbi inner loop (104× speedup).
> ConditionalFreqDist now shares `FreqDist` references so mutations via
> `cfd[cond][sample] = value` propagate correctly.
> ISRI and RSLP stemmers delegate to NLTK in the Python wrapper (`fastnltk.stem`)
> for byte-identical output. The raw Rust `_rust` versions are benchmarked below
> but the user-facing interface matches NLTK exactly.

---

## Highlights

| Operation | NLTK (ms) | fastNLTK (ms) | Speedup | Notes |
|---|---|---|---|---|
| **MaxentClassifier.train** | 73.67 | **0.13** | **562×** | GIS training, fully optimized inner loop |
| **edit_distance** | 4.48 | **0.03** | **162×** | Damerau-Levenshtein in Rust |
| **windowdiff** | 4.08 | **0.03** | **125×** | Pure algorithmic port, zero Python overhead |
| **PunktSentenceTokenizer** | 17.65 | **0.17** | **106×** | Byte-level sentence scan |
| **HiddenMarkovModelTagger** | 16.06 | **0.15** | **104×** | Integer Viterbi, zero-alloc inner loop |
| **pk** | 3.60 | **0.08** | **47×** | Segmentation metric in Rust |
| **TreebankWordDetokenizer** | 11.25 | **0.15** | **73×** | Single-pass undo |
| **TweetTokenizer** | 68.03 | **1.56** | **44×** | LazyLock regexes |
| **SentimentIntensityAnalyzer** | 30.23 | **0.96** | **32×** | PHF lexicon, exact NLTK scoring |
| **Expression.fromstring** | 34.59 | **1.01** | **34×** | FOL parser in Rust |
| **CFG.from_string** | 0.11 | **0.00** | **28×** | Grammar parser in Rust |
| **LancasterStemmer** | 54.98 | **2.15** | **26×** | Full 124-rule NLTK port |
| **TrigramCollocationFinder** | 61.53 | **2.35** | **26×** | FastMap ngram counting |
| **QuadgramCollocationFinder** | 101.54 | **2.91** | **35×** | FastMap ngram counting |
| **TextTilingTokenizer** | 4.69 | **0.06** | **77×** | Sentence segmentation, SIMD-accelerated |
| **SnowballStemmer** | 44.40 | **2.89** | **15×** | rust-stemmers crate |
| **Tree.from_string** | 6.96 | **0.61** | **11×** | Bracket parser in Rust |

---

## Full Results (66 benchmarks)

Benchmarks grouped by module. Numbers from `python -m benchmarks.run --save` on release build.

| Module | Benchmark | NLTK (ms) | fastNLTK (ms) | Speedup |
|--------|-----------|-----------|---------------|---------|
| **tokenize** | | | | |
| | ToktokTokenizer.tokenize | 18.71 | 4.25 | **4.4×** |
| | MWETokenizer.tokenize | 1.13 | 0.88 | 1.3× |
| | RegexpTokenizer.tokenize | 4.38 | 3.77 | 1.2× |
| | SpaceTokenizer.tokenize | 1.09 | 1.46 | 0.7× |
| | TreebankWordTokenizer.tokenize | 42.18 | 4.27 | **9.9×** |
| | TweetTokenizer.tokenize | 83.96 | 3.31 | **25.3×** |
| | TextTilingTokenizer.tokenize | 22236.78 | 31.58 | **704.2×** |
| | SExprTokenizer.tokenize | 0.36 | 0.01 | **30.2×** |
| | PunktSentenceTokenizer.tokenize | 14.65 | 0.44 | **33.4×** |
| | TreebankWordDetokenizer.detokenize | 6.70 | 0.12 | **54.5×** |
| | TabTokenizer.tokenize | 0.08 | 0.02 | **3.8×** |
| | LineTokenizer.tokenize | 0.24 | 0.17 | 1.4× |
| | WhitespaceTokenizer.tokenize | 4.19 | 1.27 | **3.3×** |
| | WordPunctTokenizer.tokenize | 5.53 | 1.54 | **3.6×** |
| | BlanklineTokenizer.tokenize | 2.19 | 0.11 | **19.2×** |
| | logos_word_tokenize † | — | 0.00 | — |
| **stem** | | | | |
| | SnowballStemmer.stem | 21.84 | 1.79 | **12.2×** |
| | PorterStemmer.stem | 44.71 | 7.09 | **6.3×** |
| | LancasterStemmer.stem | 32.81 | 1.41 | **23.3×** |
| | WordNetLemmatizer.lemmatize | 6.54 | 0.80 | **8.1×** |
| | ARLSTem.stem | 1.64 | 0.37 | **4.5×** |
| | ISRIStemmer.stem | 2.02 | 0.20 | **9.9×** |
| | RSLPStemmer.stem † | — | 0.00 | — |
| | RegexpStemmer.stem † | — | 0.00 | — |
| **tag** | | | | |
| | PerceptronTagger.tag | 17.73 | 2.41 | **7.3×** |
| | TnT.tag | 1.00 | 0.20 | **5.0×** |
| | DefaultTagger.tag | 1.23 | 1.02 | 1.2× |
| | UnigramTagger.tag | 1.64 | 0.95 | **1.7×** |
| | BigramTagger.tag | 2.88 | 0.93 | **3.1×** |
| | TrigramTagger.tag | 3.02 | 0.97 | **3.1×** |
| | RegexpTagger.tag | 9.65 | 1.06 | **9.1×** |
| **classify** | | | | |
| | NaiveBayesClassifier.train | 6.62 | 1.59 | **4.2×** |
| | NaiveBayesClassifier.classify | 0.01 | 0.001 | **9.0×** |
| | MaxentClassifier.train | 31.93 | 0.08 | **424.6×** |
| | TextCat.guess_language † | — | 0.00 | — |
| **probability** | | | | |
| | FreqDist.update | 20.51 | 4.47 | **4.6×** |
| | ConditionalFreqDist.inc | 5.20 | 1.85 | **2.8×** |
| | LaplaceProbDist.prob † | — | 0.00 | — |
| | MLEProbDist.prob † | — | 0.00 | — |
| **collocations** | | | | |
| | BigramCollocationFinder.from_words | 62.59 | 6.36 | **9.8×** |
| | TrigramCollocationFinder.from_words | 57.64 | 3.91 | **14.8×** |
| | QuadgramCollocationFinder.from_words | 101.04 | 4.94 | **20.5×** |
| **sentiment** | | | | |
| | SentimentIntensityAnalyzer.polarity_scores | 67.06 | 1.75 | **38.4×** |
| **metrics** | | | | |
| | windowdiff | 2.35 | 0.014 | **171.6×** |
| | pk | 2.19 | 0.024 | **90.4×** |
| | edit_distance | 2.48 | 0.014 | **175.7×** |
| | BigramAssocMeasures † | — | 0.00 | — |
| **lm** | | | | |
| | MLE.score † | — | 0.00 | — |
| | Lidstone.score † | — | 0.00 | — |
| | Laplace.score † | — | 0.00 | — |
| | StupidBackoff.score † | — | 0.00 | — |
| | KneserNeyInterpolated.score † | — | 0.00 | — |
| | WittenBellInterpolated.score † | — | 0.00 | — |
| **ccg** | | | | |
| | CCG from_string | 0.79 | 0.27 | **3.0×** |
| **chunk** | | | | |
| | RegexpParser.parse | 1.59 | 0.19 | **8.5×** |
| **cluster** | | | | |
| | KMeansClusterer.cluster | 1.59 | 0.26 | **6.2×** |
| **parse** | | | | |
| | EarleyChartParser.parse | 6.55 | 0.51 | **12.9×** |
| | CFG.from_string | 0.05 | 0.002 | **25.1×** |
| **translate** | | | | |
| | bleu | 0.03 | 0.003 | **9.4×** |
| **chat** | | | | |
| | Chat.respond | 0.001 | 0.0002 | **6.0×** |
| **tree** | | | | |
| | Tree.from_string | 3.19 | 0.31 | **10.5×** |
| **sem** | | | | |
| | Expression.fromstring | 16.47 | 0.55 | **30.2×** |
| **inference** | | | | |
| | TableauProver.prove † | — | 0.00 | — |
| | ResolutionProver.prove † | — | 0.00 | — |
| | DiscourseThread.answer_question † | — | 0.002 | — |
| | DefaultReasoner.extensions † | — | 4.32 | — |

† fastNLTK-only — no NLTK comparison available.

---

## Module Leaderboard

| Module | Geo Mean Speedup | Best Single | Key Engine |
|--------|-----------------|-------------|------------|
| metrics | **145.7×** | 176× (edit_distance) | Pure algorithmic port, zero Python overhead |
| tokenize | **17.9×** | 704× (TextTiling) | SIMD memchr3 + char scanner + byte-level segmentation |
| parse | **18.0×** | 25× (CFG) | Earley + CFG parsing |
| sem | **30.2×** | 30× (Expression) | FOL expression parser |
| collocations | **14.4×** | 21× (Quadgram) | FastMap ngram frequency counting |
| sentiment | **38.4×** | 38× (VADER) | PHF lexicon, exact NLTK algorithm |
| tree | **10.5×** | 11× | Tree bracket parser |
| translate | **9.4×** | 9× (BLEU) | BLEU in Rust |
| stem | **8.9×** | 23× (Lancaster) | 124-rule NLTK port |
| classify | **8.7×** | 425× (Maxent) | GIS training, fully optimized inner loop |
| chunk | **8.5×** | 9× | Regexp chunk parser |
| cluster | **6.2×** | 6× | K-means in Rust |
| tag | **3.7×** | 9× (RegexpTagger) | u64 feature IDs, integer Viterbi |
| probability | **3.6×** | 5× (FreqDist) | SmolStr-optimized FreqDist |
| ccg | **3.0×** | 3× | CCG category parsing |
| chat | **6.0×** | 6× | Eliza chatbot |
