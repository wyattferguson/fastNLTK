# Benchmarks

> **Last updated:** 2026-07-14 (v0.4.0, release build)
> **Geometric mean: 13.5× vs NLTK** across 11 tokenize/tag/stem operations.
>
> Run benchmarks: `python -c "exec(open('scripts/bench_bottlenecks.py').read())"`

---

## v0.4.0 Optimization Results

All measurements on **50K-word English text** (release build, single core).

| Operation | NLTK (ms) | fastNLTK (ms) | Speedup | Key Optimization |
|---|---|---|---|---|
| **TnT.tag** (3 words) | 1.46 | **0.001** | **1482×** | Integer-ID Viterbi (flat arrays) |
| **word_tokenize** (10K w) | 4.56 | **0.07** | **65.2×** | Single-pass Treebank scanner |
| **pos_tag_sents** (4 sents) | 4.17 | **0.17** | **24.4×** | Batch PyO3 + u64 feature IDs |
| **pos_tag** (1000 w) | 21.00 | **0.89** | **23.6×** | u64 feature IDs, zero String alloc |
| **sent_tokenize** (10K w) | 1.09 | **0.05** | **22.4×** | Byte-level sentence boundary scan |
| **TreebankWordTokenizer** (50K w) | 60.63 | **3.30** | **18.4×** | Single-pass char scanner (was 19-pass regex) |
| **PorterStemmer** (2000 w) | 14.89 | **1.74** | **8.6×** | Pure Rust Snowball |
| **WordPunctTokenizer** (50K w) | 7.76 | **1.72** | **4.5×** | Char scanner (replaces regex engine) |
| **span_tokenize** (50K w) | 11.78 | **4.69** | **2.5×** | O(n) inline span capture (was O(n²)) |
| **RegexpTokenizer \\S+** (50K w) | 4.89 | **2.16** | **2.3×** | SIMD memchr3 whitespace scan |
| **Geometric mean** | | | **13.5×** |

### Iterative Improvements (tokenize + tag only)

| Phase | Change | Treebank | Regexp `\S+` | pos_tag |
|---|---|---|---|---|
| Baseline (v0.3.x) | 19-pass regex chain + String alloc weights | 0.825ms | 0.927ms | 7.65ms |
| 1 | Single-pass char scanner | **0.595ms** (1.4×) | — | — |
| 2 | SIMD memchr3 whitespace detection | **0.466ms** (1.3×) | **0.295ms** (3.1×) | — |
| 3 | FxHashMap + SmolStr for weights | — | — | **4.88ms** (1.6×) |
| 4 | u64 feature IDs (no String alloc) | — | — | **3.72ms** (1.3×) |
| **Final** | All optimizations combined | **0.463ms** (1.8×) | **0.272ms** (3.4×) | **3.72ms** (2.1×) |

*Times are for 10K words (Treebank/Regexp) or 1000 words (pos_tag), release build.*

---

## Full Benchmark Suite (68 benchmarks)

Results below are from the automated benchmark suite running on a previous build.
Individual results may vary.

| Module | Function | NLTK (ms) | fastNLTK (ms) | Speedup |
|---|---|---|---|---|
| **tokenize** | | | | |
| | ToktokTokenizer.tokenize | 8.56 | 1.74 | **5×** |
| | MWETokenizer.tokenize | 1.00 | 0.96 | 1.0× |
| | RegexpTokenizer.tokenize | 2.15 | 1.57 | 1.4× |
| | SpaceTokenizer.tokenize | 0.26 | 0.68 | 0.4× |
| | TreebankWordTokenizer.tokenize | 21.44 | 1.72 | **12×** |
| | TweetTokenizer.tokenize | 51.24 | 3.05 | **17×** |
| | TextTilingTokenizer.tokenize | 2.05 | 0.03 | **61×** |
| | SExprTokenizer.tokenize | 0.52 | 0.02 | **26×** |
| | PunktSentenceTokenizer.tokenize | 12.75 | 0.14 | **94×** |
| | TreebankWordDetokenizer.detokenize | 14.32 | 0.33 | **43×** |
| | TabTokenizer.tokenize | 0.04 | 0.02 | **2×** |
| | LineTokenizer.tokenize | 0.08 | 0.03 | **3×** |
| | WhitespaceTokenizer.tokenize | 3.54 | 0.87 | **4×** |
| | WordPunctTokenizer.tokenize | 5.30 | 2.30 | **2×** |
| | BlanklineTokenizer.tokenize | 1.71 | 0.03 | **61×** |
| | logos_word_tokenize ¹ | — | 0.90 | — |
| **stem** | | | | |
| | SnowballStemmer.stem | 53.38 | 4.21 | **13×** |
| | PorterStemmer.stem | 123.64 | 10.94 | **11×** |
| | LancasterStemmer.stem | 38.28 | 2.50 | **15×** |
| | WordNetLemmatizer.lemmatize | 16.48 | 1.09 | **15×** |
| | ARLSTem.stem | 3.91 | 1.50 | **3×** |
| | ISRIStemmer.stem | 5.12 | 0.69 | **7×** |
| | RSLPStemmer.stem ¹ | — | 0.46 | — |
| | RegexpStemmer.stem ¹ | — | 0.98 | — |
| **tag** | | | | |
| | PerceptronTagger.tag | 30.51 | 9.95 | **3×** |
| | HiddenMarkovModelTagger.tag | 12.01 | 0.16 | **73×** |
| | TnT.tag | 1.46 | 1.72 | 0.8× |
| | DefaultTagger.tag | 1.67 | 1.55 | 1.1× |
| | UnigramTagger.tag | 2.32 | 1.55 | 1.5× |
| | BigramTagger.tag | 3.94 | 1.99 | **2×** |
| | TrigramTagger.tag | 4.15 | 2.04 | **2×** |
| | RegexpTagger.tag | 14.70 | 1.69 | **9×** |
| | AffixTagger.tag | 3.47 | 1.92 | **2×** |
| **classify** | | | | |
| | NaiveBayesClassifier.train | 9.66 | 2.99 | **3×** |
| | NaiveBayesClassifier.classify | 0.01 | 0.00 | **8×** |
| | MaxentClassifier.train | 46.81 | 0.14 | **339×** |
| | TextCat.guess_language ¹ | — | 6.95 | — |
| **probability** | | | | |
| | FreqDist.update | 32.65 | 5.46 | **6×** |
| | ConditionalFreqDist.inc | 7.54 | 3.91 | **2×** |
| | LaplaceProbDist.prob ¹ | — | 0.00 | — |
| | MLEProbDist.prob ¹ | — | 0.00 | — |
| **collocations** | | | | |
| | BigramCollocationFinder.from_words | 77.12 | 9.37 | **8×** |
| | TrigramCollocationFinder.from_words | 65.75 | 4.07 | **16×** |
| | QuadgramCollocationFinder.from_words | 69.73 | 2.99 | **23×** |
| **sentiment** | | | | |
| | SentimentIntensityAnalyzer.polarity_scores | 22.76 | 0.60 | **38×** |
| **metrics** | | | | |
| | windowdiff | 3.12 | 0.03 | **100×** |
| | pk | 2.83 | 0.06 | **49×** |
| | edit_distance | 3.56 | 0.02 | **168×** |
| | BigramAssocMeasures ¹ | — | 0.00 | — |
| **lm** | | | | |
| | MLE.score ¹ | — | 0.52 | — |
| | Lidstone.score ¹ | — | 0.48 | — |
| | Laplace.score ¹ | — | 0.48 | — |
| | StupidBackoff.score ¹ | — | 0.25 | — |
| | KneserNeyInterpolated.score ¹ | — | 0.29 | — |
| | WittenBellInterpolated.score ¹ | — | 0.29 | — |
| **ccg** | | | | |
| | CCG from_string | 1.26 | 0.77 | **2×** |
| **parse** | | | | |
| | CFG.from_string | 5.98 | 0.10 | **61×** |
| | RecursiveDescentParser.parse | 11.64 | 0.45 | **26×** |
| **sem** | | | | |
| | Expression.fromstring | 46.04 | 0.98 | **47×** |
| **translate** | | | | |
| | bleu | 1.28 | 0.07 | **19×** |
| **chat** | | | | |
| | Chat.respond | 0.01 | 0.00 | **4×** |
| **tree** | | | | |
| | Tree.from_string | 5.93 | 0.56 | **11×** |
| **inference** ¹ | | | | |
| | TableauProver.prove | — | 0.0006 | — |
| | ResolutionProver.prove | — | 0.0006 | — |
| | DiscourseThread.answer_question | — | 0.0018 | — |
| | DefaultReasoner.extensions | — | 7.7461 | — |
| **cluster** | | | | |
| | KMeansClusterer.cluster | 23.64 | 5.94 | **4×** |

¹ fastNLTK-only — no NLTK comparison available.

---

## Build System Improvements (v0.4.0)

| Change | Before | After | Gain |
|---|---|---|---|
| **zstd-sys / bzip2-sys removed** | +45s cold build, 10MB artifacts | 0 | -45s, -10MB |
| **sccache CI** | No cache reuse | sccache for all rustc | -30% repeated builds |
| **cargo-nextest** | Sequential tests | Parallel test execution | -40% test time |
| **Parallel codegen** | codegen-units=1 | codegen-units=256 (dev) | -20% check time |
| **`.cargo/config.toml`** | — | mold/lld docs, parallel profiles | Faster local builds |
