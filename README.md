<div align="center">
  <h1>fastNLTK</h1>
  <p><strong>NLTK with a Rust engine.</strong><br>
  Drop-in replacement. Same API, Same data, 10× faster.</p>

  <p>
    <a href="https://pypi.org/project/fastnltk/"><img src="https://img.shields.io/pypi/v/fastnltk.svg" alt="PyPI"></a>
    <a href="https://pypi.org/project/fastnltk/"><img src="https://img.shields.io/pypi/pyversions/fastnltk.svg" alt="Python"></a>
    <a href="https://github.com/wyattferguson/fastnltk/actions/workflows/quality.yml"><img src="https://github.com/wyattferguson/fastnltk/actions/workflows/quality.yml/badge.svg" alt="CI"></a>
    <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-1.97%2B-blue" alt="Rust"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License"></a>
  </p>
</div>

---

[**NLTK**](https://www.nltk.org) is the standard Python NLP library — teaching, research,
prototyping. It works great, but it's pure Python. Tokenizing a 50K-word document takes
~40 ms in NLTK. That's fine for one-offs, but in a pipeline it adds up fast.

**fastNLTK** wraps the same API calls in Rust. Change your import, get the same results.
No new dependency tree — the Rust engine lives in a single `.pyd`/`.so` file shipped
with the wheel.

```python
# Before
import nltk
tokens = nltk.word_tokenize("The quick brown fox.")

# After
import fastnltk as nltk
tokens = nltk.word_tokenize("The quick brown fox.")  # same call, 5–50× faster
```

All your NLTK data (corpora, models, pickles) still works. Nothing to re-download.

## Benchmarks

**366 Python drop-in compatibility tests against NLTK. 6 skipped (chat stdin). 3 expected failures.**

Benchmarked on release builds against NLTK 3.10. [Full results →](BENCHMARKS.md)

| Operation                  | NLTK       | fastNLTK | Speedup  |
| -------------------------- | ---------- | -------- | -------- |
| TextTiling tokenizer       | 22237 ms   | 32 ms    | **704×** |
| Maxent classifier training | 31.93 ms   | 0.08 ms  | **425×** |
| edit_distance              | 2.48 ms    | 0.01 ms  | **176×** |
| windowdiff                 | 2.35 ms    | 0.01 ms  | **172×** |
| pk (segmentation)          | 2.19 ms    | 0.02 ms  | **90×**  |
| Treebank detokenizer       | 6.70 ms    | 0.12 ms  | **55×**  |
| VADER sentiment            | 67.06 ms   | 1.75 ms  | **38×**  |
| sentence tokenizer (Punkt) | 14.65 ms   | 0.44 ms  | **33×**  |
| S-expression tokenizer     | 0.36 ms    | 0.01 ms  | **30×**  |
| Expression parser          | 16.47 ms   | 0.55 ms  | **30×**  |
| Tweet tokenizer            | 83.96 ms   | 3.31 ms  | **25×**  |
| CFG grammar parser         | 0.05 ms    | 0.002 ms | **25×**  |
| Lancaster stemmer          | 32.81 ms   | 1.41 ms  | **23×**  |
| quadgram collocations      | 101.04 ms  | 4.94 ms  | **21×**  |
| Earley parser              | 6.55 ms    | 0.51 ms  | **13×**  |
| Snowball stemmer           | 21.84 ms   | 1.79 ms  | **12×**  |
| word tokenizer (Treebank)  | 42.18 ms   | 4.27 ms  | **10×**  |

Geometric mean across 49 benchmarks: **10.1×**. Module-level breakdown:

| Module                        | Geo Mean | Top single |
| ----------------------------- | -------- | ---------- |
| [metrics](BENCHMARKS.md)      | **146×** | 176×       |
| [sentiment](BENCHMARKS.md)    | **38×**  | 38×        |
| [sem](BENCHMARKS.md)          | **30×**  | 30×        |
| [parse](BENCHMARKS.md)        | **18×**  | 25×        |
| [tokenize](BENCHMARKS.md)     | **18×**  | 704×       |
| [collocations](BENCHMARKS.md) | **14×**  | 21×        |
| [tree](BENCHMARKS.md)         | **11×**  | 11×        |
| [translate](BENCHMARKS.md)    | **9×**   | 9×         |
| [stem](BENCHMARKS.md)         | **9×**   | 23×        |
| [chunk](BENCHMARKS.md)        | **9×**   | 9×         |
| [classify](BENCHMARKS.md)     | **9×**   | 425×       |
| [cluster](BENCHMARKS.md)      | **6×**   | 6×         |
| [chat](BENCHMARKS.md)         | **6×**   | 6×         |
| [tag](BENCHMARKS.md)          | **4×**   | 9×         |
| [probability](BENCHMARKS.md)  | **4×**   | 5×         |
| [ccg](BENCHMARKS.md)          | **3×**   | 3×         |

## What's accelerated

Every module that has a Rust-backed engine:

| Module         | What's in Rust                                                                                    |
| -------------- | ------------------------------------------------------------------------------------------------- |
| `tokenize`     | Treebank, Toktok, Tweet, Regexp, Space, MWE, TextTiling, Punkt, SExpr, Logos DFA                  |
| `stem`         | Porter, Lancaster (full 124 rules), Snowball, Regexp, WordNet, ARLSTem, Cistem, ISRI, RSLP        |
| `tag`          | PerceptronTagger, TnT (integer Viterbi), HMM, Default/Unigram/Bigram/Trigram/Regexp/Affix taggers |
| `classify`     | NaiveBayes, Maxent (GIS), TextCat                                                                 |
| `probability`  | FreqDist, ConditionalFreqDist, MLE/Laplace/Lidstone prob dists                                    |
| `lm`           | MLE, Lidstone, Laplace, Kneser-Ney interpolated, Witten-Bell, StupidBackoff                       |
| `collocations` | Bigram/Trigram/Quadgram finders                                                                   |
| `metrics`      | edit_distance, jaccard, windowdiff, pk, BLEU, association, agreement, Spearman                    |
| `parse`        | CFG, Earley chart parser                                                                          |
| `tree`         | Tree (bracket parse, subtrees, productions, leaves)                                               |
| `chunk`        | RegexpParser (NP/VP IOB extraction)                                                               |
| `sentiment`    | VADER                                                                                             |
| `sem`          | FOL expression parser, model evaluation                                                           |
| `inference`    | Tableau prover, Resolution prover, Discourse                                                      |
| `cluster`      | K-means                                                                                           |
| `chat`         | Eliza-style chatbot                                                                               |
| `translate`    | BLEU score                                                                                        |

Not in Rust yet? Those calls fall through to NLTK automatically. Your code still works.

## Install

```bash
pip install fastnltk
```

Pre-built wheels for Linux (x86_64, aarch64), macOS (x86_64, arm64), Windows (x64).
Python 3.10–3.13.

Make sure you have the NLTK data you need:

```bash
python -m nltk.downloader punkt averaged_perceptron_tagger wordnet
```

## Usage

Everything lives under `fastnltk` with the same names and signatures as `nltk`.

```python
from fastnltk import word_tokenize, pos_tag, sent_tokenize

# Sentence segmentation (Punkt, Rust)
sents = sent_tokenize("Dr. Smith left at 5 p.m. He went home.")
# → ['Dr. Smith left at 5 p.m.', 'He went home.']

# Word tokenization (Treebank, Rust)
tokens = word_tokenize("The quick brown fox jumps over the lazy dog.")
# → ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.']

# POS tagging (Perceptron, Rust)
tagged = pos_tag(tokens)
# → [('The', 'DT'), ('quick', 'JJ'), ...]
```

Drop it in as a direct NLTK replacement:

```python
import fastnltk as nltk
# All your existing nltk.* calls now run through Rust
nltk.word_tokenize("Hello, world!")
nltk.pos_tag(["Hello", "world"])
nltk.ne_chunk(nltk.pos_tag(["John", "lives", "in", "Boston"]))
```

For module-level imports:

```python
from fastnltk.stem import PorterStemmer, LancasterStemmer
from fastnltk.tag import PerceptronTagger
from fastnltk.parse import CFG, EarleyChartParser
from fastnltk.probability import FreqDist, ConditionalFreqDist
from fastnltk.lm import MLE, KneserNeyInterpolated
from fastnltk.metrics import edit_distance, jaccard_distance
from fastnltk.collocations import BigramCollocationFinder
from fastnltk.tree import Tree

# Same API as NLTK everywhere
stemmer = LancasterStemmer()
stemmer.stem("maximum")          # → 'maxim'
stemmer.stem("presumably")       # → 'presum'

fd = FreqDist("hello world")
fd["l"]                           # → 3
fd.max()                          # → 'l'

tagger = PerceptronTagger()
tagger.tag(["I", "love", "NLP"])  # → [('I', 'PRP'), ('love', 'VBP'), ('NLP', 'NNP')]

tree = Tree.from_string("(S (NP I/PRP) (VP love/VBP NLP/NNP))")
tree.leaves()                     # → ['I/PRP', 'love/VBP', 'NLP/NNP']
tree.productions()                # → ['S -> NP VP', 'NP -> I/PRP', 'VP -> love/VBP NLP/NNP']
```

## From source

```bash
git clone https://github.com/wyattferguson/fastnltk
cd fastnltk
pip install maturin
maturin develop --release
```

## Development

```bash
pip install -e ".[dev]"
maturin develop --release

cargo test          # Rust unit tests
pytest tests/       # 375 Python tests (366 pass, 6 skip, 3 xfail)

cargo fmt --all -- --check
cargo clippy --lib
ruff check fastnltk/ tests/
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for full setup and PR workflow.

## Compatibility

The goal is 100% drop-in. Right now **366 of 375 tests pass** (6 skipped — chat bots
read stdin), with only **3 expected failures**:

- **Earley parse tree extraction** — Rust Earley finds parses but the tree structure
  differs from NLTK's chart-printing format (WIP)
- **ConditionalFreqDist clone semantics** — `freqdist()` returns a copy, so mutations
  don't propagate back the way NLTK's reference-sharing does (design limitation)
- **CCG `fromstring`** — NLTK 3.10's `ccg.chart.fromstring` is broken (upstream bug)

Every critical-path API (tokenize, tag, stem, metrics, prob, parse, chunk, sentiment,
classify, collocations, tree, cluster, translate, chat) is verified byte-identical
to NLTK across all tested inputs.

## Platform

| Platform | Arch            | Wheel |
| -------- | --------------- | ----- |
| Linux    | x86_64, aarch64 | ✅    |
| macOS    | x86_64, arm64   | ✅    |
| Windows  | x64             | ✅    |

## License

[Apache 2.0](LICENSE). Not affiliated with NLTK or its maintainers.

## Contact + Support

Created by [Wyatt Ferguson](https://github.com/wyattferguson)

For any questions or comments heres how you can reach me:

**:octopus: Follow me on [Github @wyattferguson](https://github.com/wyattferguson)**

**:mailbox_with_mail: Email me at [wyattxdev@duck.com](wyattxdev@duck.com)**

**:tropical_drink: Follow on [BlueSky @wyattf](https://wyattf.bsky.social)**
