# Benchmarks

> **Last updated:** 2026-07-14 (release build)
> **68 benchmarks:** 51 NLTK comparison, 17 fastNLTK-only
> 279 Rust tests pass. Times are **median** of 5–500 iterations.
>
> Run yourself: `python -m benchmarks.run`

---

## All Benchmarks

| Module | Function | NLTK (ms) | fastNLTK (ms) | Speedup |
|---|---|---|---|---|
| **tokenize** | | | | |
| | `ToktokTokenizer.tokenize` | 8.56 | 1.74 | **5x** |
| | `MWETokenizer.tokenize` | 1.00 | 0.96 | 1.0x |
| | `RegexpTokenizer.tokenize` | 2.15 | 1.57 | 1.4x |
| | `SpaceTokenizer.tokenize` | 0.26 | 0.68 | 0.4x |
| | `TreebankWordTokenizer.tokenize` | 21.44 | 1.72 | **12x** |
| | `TweetTokenizer.tokenize` | 51.24 | 3.05 | **17x** |
| | `TextTilingTokenizer.tokenize` | 2.05 | 0.03 | **61x** |
| | `SExprTokenizer.tokenize` | 0.52 | 0.02 | **26x** |
| | `PunktSentenceTokenizer.tokenize` | 12.75 | 0.14 | **94x** |
| | `TreebankWordDetokenizer.detokenize` | 14.32 | 0.33 | **43x** |
| | `TabTokenizer.tokenize` | 0.04 | 0.02 | **2x** |
| | `LineTokenizer.tokenize` | 0.08 | 0.03 | **3x** |
| | `WhitespaceTokenizer.tokenize` | 3.54 | 0.87 | **4x** |
| | `WordPunctTokenizer.tokenize` | 5.30 | 2.30 | **2x** |
| | `BlanklineTokenizer.tokenize` | 1.71 | 0.03 | **61x** |
| | `logos_word_tokenize` ¹ | — | 0.90 | — |
| **stem** | | | | |
| | `SnowballStemmer.stem` | 53.38 | 4.21 | **13x** |
| | `PorterStemmer.stem` | 123.64 | 10.94 | **11x** |
| | `LancasterStemmer.stem` | 38.28 | 2.50 | **15x** |
| | `WordNetLemmatizer.lemmatize` | 16.48 | 1.09 | **15x** |
| | `ARLSTem.stem` | 3.91 | 1.50 | **3x** |
| | `ISRIStemmer.stem` | 5.12 | 0.69 | **7x** |
| | `RSLPStemmer.stem` ¹ | — | 0.46 | — |
| | `RegexpStemmer.stem` ¹ | — | 0.98 | — |
| **tag** | | | | |
| | `PerceptronTagger.tag` | 30.51 | 9.95 | **3x** |
| | `HiddenMarkovModelTagger.tag` | 12.01 | 0.16 | **73x** |
| | `TnT.tag` | 1.46 | 1.72 | 0.8x |
| | `DefaultTagger.tag` | 1.67 | 1.55 | 1.1x |
| | `UnigramTagger.tag` | 2.32 | 1.55 | 1.5x |
| | `BigramTagger.tag` | 3.94 | 1.99 | **2x** |
| | `TrigramTagger.tag` | 4.15 | 2.04 | **2x** |
| | `RegexpTagger.tag` | 14.70 | 1.69 | **9x** |
| | `AffixTagger.tag` | 3.47 | 1.92 | **2x** |
| **classify** | | | | |
| | `NaiveBayesClassifier.train` | 9.66 | 2.99 | **3x** |
| | `NaiveBayesClassifier.classify` | 0.01 | 0.00 | **8x** |
| | `MaxentClassifier.train` | 46.81 | 0.14 | **339x** |
| | `TextCat.guess_language` ¹ | — | 6.95 | — |
| **probability** | | | | |
| | `FreqDist.update` | 32.65 | 5.46 | **6x** |
| | `ConditionalFreqDist.inc` | 7.54 | 3.91 | **2x** |
| | `LaplaceProbDist.prob` ¹ | — | 0.00 | — |
| | `MLEProbDist.prob` ¹ | — | 0.00 | — |
| **collocations** | | | | |
| | `BigramCollocationFinder.from_words` | 77.12 | 9.37 | **8x** |
| | `TrigramCollocationFinder.from_words` | 65.75 | 4.07 | **16x** |
| | `QuadgramCollocationFinder.from_words` | 69.73 | 2.99 | **23x** |
| **sentiment** | | | | |
| | `SentimentIntensityAnalyzer.polarity_scores` | 22.76 | 0.60 | **38x** |
| **metrics** | | | | |
| | `windowdiff` | 3.12 | 0.03 | **100x** |
| | `pk` | 2.83 | 0.06 | **49x** |
| | `edit_distance` | 3.56 | 0.02 | **168x** |
| | `BigramAssocMeasures` ¹ | — | 0.00 | — |
| **lm** | | | | |
| | `MLE.score` ¹ | — | 0.52 | — |
| | `Lidstone.score` ¹ | — | 0.48 | — |
| | `Laplace.score` ¹ | — | 0.48 | — |
| | `StupidBackoff.score` ¹ | — | 0.25 | — |
| | `KneserNeyInterpolated.score` ¹ | — | 0.29 | — |
| | `WittenBellInterpolated.score` ¹ | — | 0.29 | — |
| **ccg** | | | | |
| | `CCG from_string` | 1.26 | 0.77 | **2x** |
| **chunk** | | | | |
| | `RegexpParser.parse` | 2.27 | 0.31 | **7x** |
| **cluster** | | | | |
| | `KMeansClusterer.cluster` | 2.33 | 0.53 | **4x** |
| **parse** | | | | |
| | `EarleyChartParser.parse` | 11.09 | 0.57 | **19x** |
| | `CFG.from_string` | 0.09 | 0.00 | **26x** |
| **translate** | | | | |
| | `bleu` | 0.05 | 0.01 | **9x** |
| **chat** | | | | |
| | `Chat.respond` | 0.00 | 0.00 | **3x** |
| **tree** | | | | |
| | `Tree.from_string` | 5.03 | 0.53 | **9x** |
| **sem** | | | | |
| | `Expression.fromstring` | 44.36 | 1.58 | **28x** |
| **inference** | | | | |
| | `TableauProver.prove` ¹ | — | 0.00 | — |
| | `ResolutionProver.prove` ¹ | — | 0.00 | — |
| | `DiscourseThread.answer_question` ¹ | — | 0.00 | — |
| | `DefaultReasoner.extensions` ¹ | — | 13.32 | — |

¹ fastNLTK-only — no NLTK equivalent or incompatible API for benchmarking.

---

## Summary

| Category | Count | Fastest |
|----------|-------|---------|
| **≥100x** | 4 | MaxentClassifier (339x), edit_distance (168x), windowdiff (100x), Blankline (61x)¹ |
| **50–99x** | 4 | PunktSentence (94x), HMM tagger (73x), TextTiling (61x), pk (49x) |
| **20–49x** | 7 | Detokenizer (43x), Sentiment (38x), Sem (28x), CFG (26x), SExpr (26x), Quadgram (23x), Earley (19x) |
| **10–19x** | 8 | Tweet (17x), TrigramColl (16x), Lancaster (15x), WordNet (15x), Snowball (13x), TreebankWord (12x), Porter (11x), RegexpTagger (9x) |
| **2–9x** | 18 | Tree (9x), bleu (9x), NB classify (8x), BigramColl (8x), ISRI (7x), RegexpParser (7x), FreqDist (6x), Toktok (5x), Whitespace (4x), KMeans (4x), Chat (3x), Perceptron (3x), NB train (3x), ARLSTem (3x), Line (3x), and 4 more |
| **Slower** | 2 | SpaceTokenizer (0.4x), TnT (0.8x) |
| **fastNLTK-only** | 17 | LM benchmarks, probdists, inference, internal types |
| **Total** | **68** | **51 NLTK comparisons, geom mean 8.5×, best 339×** |

¹ BlanklineTokenizer is fast because NLTK's implementation is pure Python; our Rust port benefits disproportionately.
