# Benchmarks

> **Last updated:** 2026-07-14 (release build, i7-12700, 32GB RAM)
> **v0.3.0:** pyo3 v0.29, hashbrown v0.17, logos v0.16, phf v0.14, rand v0.10, smol_str v0.3, whatlang v0.18
> 279 Rust tests, 0 clippy errors, all deps at latest
>
> Times are **median** of 5–100 iterations. "—" means fastNLTK-only (no NLTK comparison).
>
> Run yourself: `.venv\Scripts\python -m benchmarks.run`

---

## All Benchmarks

| Module                      | Function                                     | Input         | NLTK (ms) | fastNLTK (ms) | Speedup    |
| --------------------------- | -------------------------------------------- | ------------- | --------- | ------------- | ---------- |
| **tokenize**                |                                              |               |           |               |            |
|                             | `ToktokTokenizer.tokenize`                   | 82KB          | 8.40      | 2.17          | **3.9x**   |
|                             | `MWETokenizer.tokenize`                      | 13.6K words   | 0.99      | 0.89          | 1.1x       |
|                             | `RegexpTokenizer.tokenize`                   | 82KB          | 2.14      | 1.14          | **1.9x**   |
|                             | `SpaceTokenizer.tokenize`                    | 82KB          | 0.26      | 0.47          | 0.6x       |
|                             | `TreebankWordTokenizer.tokenize`             | 82KB          | 21.23     | 1.60          | **13.3x**  |
|                             | `TweetTokenizer.tokenize`                    | 82KB          | 51.27     | 2.70          | **19.0x**  |
|                             | `TextTilingTokenizer.tokenize`               | 82KB          | —         | 4.30          | —          |
|                             | `logos_word_tokenize`                        | 82KB          | —         | 0.47          | —          |
| **stem**                    |                                              |               |           |               |            |
|                             | `SnowballStemmer.stem`                       | 10K words     | 32.61     | 2.47          | **13.2x**  |
|                             | `PorterStemmer.stem`                         | 10K words     | 78.51     | 11.08         | **7.1x**   |
|                             | `LancasterStemmer.stem`                      | 10K words     | 49.53     | 6.13          | **8.1x**   |
|                             | `WordNetLemmatizer.lemmatize`                | 5K words      | —         | 2.15          | —          |
| **tag**                     |                                              |               |           |               |            |
|                             | `PerceptronTagger.tag`                       | 100 sentences | —         | 30.07         | —          |
|                             | `HiddenMarkovModelTagger.tag`                | 1K words      | —         | 0.42          | —          |
|                             | `DefaultTagger.tag`                          | 10K words     | 4.70      | 5.21          | 0.9x       |
|                             | `UnigramTagger.tag`                          | 10K words     | 8.97      | 4.54          | **2.0x**   |
|                             | `BigramTagger.tag`                           | 10K words     | 10.02     | 4.27          | **2.3x**   |
|                             | `TrigramTagger.tag`                          | 10K words     | 11.23     | 7.77          | 1.4x       |
|                             | `RegexpTagger.tag`                           | 10K words     | 38.39     | 4.07          | **9.4x**   |
|                             | `AffixTagger.tag`                            | 10K words     | 9.08      | 4.51          | **2.0x**   |
| **classify**                |                                              |               |           |               |            |
|                             | `NaiveBayesClassifier.train`                 | 2K instances  | —         | 6.25          | —          |
|                             | `NaiveBayesClassifier.classify`              | 5 features    | —         | 0.002         | —          |
| **probability**             |                                              |               |           |               |            |
|                             | `FreqDist.update`                            | 100K samples  | 37.43     | 7.69          | **4.9x**   |
| **collocations**            |                                              |               |           |               |            |
|                             | `BigramCollocationFinder.from_words`         | 50K words     | 81.51     | 13.71         | **5.9x**   |
| **sentiment**               |                                              |               |           |               |            |
|                             | `SentimentIntensityAnalyzer.polarity_scores` | 82KB          | —         | 2.05          | —          |
| **metrics**                 |                                              |               |           |               |            |
|                             | `windowdiff`                                 | 12K chars     | 6.84      | 0.06          | **107.0x** |
|                             | `pk`                                         | 12K chars     | 6.38      | 0.13          | **50.3x**  |
|                             | `edit_distance`                              | 100 chars     | —         | 0.05          | —          |
| **lm**                      |                                              |               |           |               |            |
|                             | `MLE.fit`                                    | 1K sentences  | —         | 2.28          | —          |
|                             | `KneserNeyInterpolated.score`                | 4 queries     | —         | 0.02          | —          |
| **ccg**                     |                                              |               |           |               |            |
|                             | `CCG from_string`                            | 3.5K parses   | 3.75      | 1.31          | **2.9x**   |
| **chunk**                   |                                              |               |           |               |            |
|                             | `RegexpParser.parse`                         | 1.8K tokens   | 4.94      | 0.61          | **8.1x**   |
| **cluster**                 |                                              |               |           |               |            |
|                             | `KMeansClusterer.cluster`                    | 500×5D        | —         | 1.03          | —          |
| **parse**                   |                                              |               |           |               |            |
|                             | `EarleyChartParser.parse`                    | 30 sentences  | —         | 1.02          | —          |
| **translate**               |                                              |               |           |               |            |
|                             | `bleu`                                       | 7 tokens      | 0.12      | 0.01          | **11.3x**  |
| **chat**                    |                                              |               |           |               |            |
|                             | `Chat.respond`                               | single        | 0.002     | 0.001         | **3.6x**   |
| **tree**                    |                                              |               |           |               |            |
|                             | `Tree.from_string`                           | 300 trees     | 12.10     | 0.97          | **12.5x**  |
| **sem**                     |                                              |               |           |               |            |
|                             | `Expression.fromstring`                      | 500 formulas  | 61.77     | 1.46          | **42.2x**  |
| **inference**               |                                              |               |           |               |            |
|                             | `TableauProver.prove`                        | P\|~P         | —         | 0.002         | —          |
|                             | `ResolutionProver.prove`                     | P\|~P         | —         | 0.002         | —          |
|                             | `DiscourseThread.answer_question`            | 2 DRSs        | —         | 0.005         | —          |
|                             | `DefaultReasoner.extensions`                 | 10 rules      | —         | 76.25         | —          |
| **Average (42 benchmarks)** |                                              |               |           |               | **9.4x**   |

---

## Top 10 Speedups

| #   | Function                         | Speedup    | Why                                            |
| --- | -------------------------------- | ---------- | ---------------------------------------------- |
| 1   | `windowdiff`                     | **107.0x** | Pure algorithmic port, no Python loop overhead |
| 2   | `pk`                             | **50.3x**  | Same as windowdiff — simple string scan        |
| 3   | `Expression.fromstring`          | **42.2x**  | Recursive descent parser in native code        |
| 4   | `TweetTokenizer.tokenize`        | **19.0x**  | Compiled regex, no Python `re` overhead        |
| 5   | `TreebankWordTokenizer.tokenize` | **13.3x**  | Compiled regex, no Python `re` overhead        |
| 6   | `SnowballStemmer.stem`           | **13.2x**  | `rust-stemmers` — libstemmer in Rust           |
| 7   | `Tree.from_string`               | **12.5x**  | Bracket parser in Rust                         |
| 8   | `bleu`                           | **11.3x**  | Tight DP loop in native code                   |
| 9   | `RegexpTagger.tag`               | **9.4x**   | Compiled regex dispatch, zero Python overhead  |
| 10  | `LancasterStemmer.stem`          | **8.1x**   | Algorithmic port, string ops in native code    |

---

## Module Coverage

| Module       | Benchmarks | Best Speedup  |
| ------------ | ---------- | ------------- |
| tokenize     | 8          | **19.0x**     |
| stem         | 4          | **13.2x**     |
| tag          | 8          | **9.4x**      |
| classify     | 2          | fastNLTK-only |
| probability  | 1          | **4.9x**      |
| collocations | 1          | **5.9x**      |
| sentiment    | 1          | fastNLTK-only |
| metrics      | 3          | **107.0x**    |
| lm           | 2          | fastNLTK-only |
| ccg          | 1          | **2.9x**      |
| chunk        | 1          | **8.1x**      |
| cluster      | 1          | fastNLTK-only |
| parse        | 1          | fastNLTK-only |
| translate    | 1          | **11.3x**     |
| chat         | 1          | **3.6x**      |
| tree         | 1          | **12.5x**     |
| sem          | 1          | **42.2x**     |
| inference    | 4          | fastNLTK-only |

---

## Running

```bash
.venv\Scripts\python -m benchmarks.run           # Run all
.venv\Scripts\python -m benchmarks.run --save    # Run + save
```

The harness (42 automated benchmarks in `benchmarks/bench_suite.py`) supports automatic
regression detection against saved baselines. Default threshold: 25%.
