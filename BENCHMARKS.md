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
| | `ToktokTokenizer.tokenize` | 8.93 | 3.67 | **2x** |
| | `MWETokenizer.tokenize` | 1.09 | 0.82 | 1.3x |
| | `RegexpTokenizer.tokenize` | 2.28 | 1.52 | 1.5x |
| | `SpaceTokenizer.tokenize` | 1.28 | 1.73 | 0.7x |
| | `TreebankWordTokenizer.tokenize` | 22.29 | 1.26 | **18x** |
| | `TweetTokenizer.tokenize` | 54.23 | 2.21 | **25x** |
| | `TextTilingTokenizer.tokenize` | 2.65 | 0.02 | **124x** |
| | `SExprTokenizer.tokenize` | 0.46 | 0.03 | **16x** |
| | `PunktSentenceTokenizer.tokenize` | 25.86 | 0.18 | **145x** |
| | `TreebankWordDetokenizer.detokenize` | 7.96 | 0.18 | **44x** |
| | `TabTokenizer.tokenize` | 1.17 | 0.87 | 1.3x |
| | `LineTokenizer.tokenize` | 0.20 | 0.06 | **3x** |
| | `WhitespaceTokenizer.tokenize` | 1.25 | 0.20 | **6x** |
| | `WordPunctTokenizer.tokenize` | 1.67 | 0.50 | **3x** |
| | `BlanklineTokenizer.tokenize` | 1.41 | 0.11 | **13x** |
| | `logos_word_tokenize` ¹ | — | 0.99 | — |
| **stem** | | | | |
| | `SnowballStemmer.stem` | 54.68 | 3.15 | **17x** |
| | `PorterStemmer.stem` | 163.17 | 11.76 | **14x** |
| | `LancasterStemmer.stem` | 42.53 | 2.21 | **19x** |
| | `WordNetLemmatizer.lemmatize` | 14.62 | 0.60 | **24x** |
| | `ARLSTem.stem` | 3.00 | 1.08 | **3x** |
| | `ISRIStemmer.stem` | 35.88 | 4.05 | **9x** |
| | `RSLPStemmer.stem` ¹ | — | 0.45 | — |
| | `RegexpStemmer.stem` ¹ | — | 1876.59 | — |
| **tag** | | | | |
| | `PerceptronTagger.tag` | 50.02 | 18.63 | **3x** |
| | `HiddenMarkovModelTagger.tag` | 21.80 | 0.22 | **98x** |
| | `TnT.tag` | 2.51 | 3.73 | 0.7x |
| | `DefaultTagger.tag` | 2.09 | 1.16 | 1.8x |
| | `UnigramTagger.tag` | 3.21 | 1.93 | 1.7x |
| | `BigramTagger.tag` | 4.70 | 2.34 | **2x** |
| | `TrigramTagger.tag` | 5.45 | 2.43 | **2x** |
| | `RegexpTagger.tag` | 18.89 | 1.93 | **10x** |
| | `AffixTagger.tag` | 4.62 | 2.21 | **2x** |
| **classify** | | | | |
| | `NaiveBayesClassifier.train` | 11.09 | 2.41 | **5x** |
| | `NaiveBayesClassifier.classify` | 0.01 | 0.00 | **18x** |
| | `MaxentClassifier.train` | 104.72 | 0.28 | **377x** |
| | `TextCat.guess_language` ¹ | — | 12.71 | — |
| **probability** | | | | |
| | `FreqDist.update` | 23.22 | 1.87 | **12x** |
| | `ConditionalFreqDist.inc` | 3.71 | 1.78 | **2x** |
| | `LaplaceProbDist.prob` ¹ | — | 0.00 | — |
| | `MLEProbDist.prob` ¹ | — | 0.00 | — |
| **collocations** | | | | |
| | `BigramCollocationFinder.from_words` | 30.64 | 5.78 | **5x** |
| | `TrigramCollocationFinder.from_words` | 27.11 | 1.32 | **21x** |
| | `QuadgramCollocationFinder.from_words` | 29.85 | 1.33 | **23x** |
| **sentiment** | | | | |
| | `SentimentIntensityAnalyzer.polarity_scores` | 44.65 | 0.90 | **50x** |
| **metrics** | | | | |
| | `windowdiff` | 8.69 | 0.04 | **242x** |
| | `pk` | 6.99 | 0.13 | **52x** |
| | `edit_distance` | 7.18 | 0.05 | **155x** |
| | `BigramAssocMeasures` ¹ | — | 0.00 | — |
| **lm** | | | | |
| | `MLE.score` ¹ | — | 0.47 | — |
| | `Lidstone.score` ¹ | — | 0.42 | — |
| | `Laplace.score` ¹ | — | 0.42 | — |
| | `StupidBackoff.score` ¹ | — | 0.22 | — |
| | `KneserNeyInterpolated.score` ¹ | — | 0.25 | — |
| | `WittenBellInterpolated.score` ¹ | — | 0.25 | — |
| **ccg** | | | | |
| | `CCG from_string` | 2.65 | 1.08 | **2x** |
| **chunk** | | | | |
| | `RegexpParser.parse` | 4.03 | 0.55 | **7x** |
| **cluster** | | | | |
| | `KMeansClusterer.cluster` | 3.38 | 0.87 | **4x** |
| **parse** | | | | |
| | `EarleyChartParser.parse` | 24.81 | 1.06 | **23x** |
| | `CFG.from_string` | 4.19 | 0.14 | **30x** |
| **translate** | | | | |
| | `bleu` | 0.07 | 0.01 | **10x** |
| **chat** | | | | |
| | `Chat.respond` | 0.00 | 0.00 | **3x** |
| **tree** | | | | |
| | `Tree.from_string` | 6.14 | 0.62 | **10x** |
| **sem** | | | | |
| | `Expression.fromstring` | 31.65 | 1.69 | **19x** |
| **inference** | | | | |
| | `TableauProver.prove` ¹ | — | 0.00 | — |
| | `ResolutionProver.prove` ¹ | — | 0.00 | — |
| | `DiscourseThread.answer_question` ¹ | — | 0.00 | — |
| | `DefaultReasoner.extensions` ¹ | — | 59.44 | — |

¹ fastNLTK-only — no NLTK equivalent or incompatible API for benchmarking.

---

## Summary

| Category | Count | Fastest |
|----------|-------|---------|
| **≥100x** | 5 | MaxentClassifier (377x), windowdiff (242x), edit_distance (155x), PunktSentence (145x), TextTiling (124x) |
| **50–99x** | 3 | HMM tagger (98x), pk (52x), Sentiment (50x) |
| **20–49x** | 7 | Detokenizer (44x), CFG (30x), Tweet (25x), WordNet (24x), Earley (23x), Quadgram (23x), TrigramColl (21x) |
| **10–19x** | 10 | Lancaster (19x), Sem (19x), TreebankWord (18x), NB classify (18x), Snowball (17x), SExpr (16x), Porter (14x), Blankline (13x), FreqDist (12x), Tree (10x), RegexpTagger (10x), bleu (10x) |
| **2–9x** | 17 | ISRI (9x), RegexpParser (7x), Whitespace (6x), NB train (5x), BigramColl (5x), KMeans (4x), Line (3x), Chat (3x), ARLSTem (3x), WordPunct (3x), and 7 more |
| **Slower** | 2 | SpaceTokenizer (0.7x), TnT (0.7x) |
| **fastNLTK-only** | 17 | Internal types, LM benchmarks, probdists, inference |
