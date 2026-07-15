<div align="center">
  <h1>fastNLTK</h1>
  <p><strong>Drop-in Rust NLTK. Same API, way faster.</strong></p>

[![PyPI version](https://img.shields.io/pypi/v/fastnltk.svg)](https://pypi.org/project/fastnltk/)
[![Python](https://img.shields.io/pypi/pyversions/fastnltk.svg)](https://pypi.org/project/fastnltk/)
[![CI](https://github.com/fastnltk/fastnltk/actions/workflows/quality.yml/badge.svg)](https://github.com/fastnltk/fastnltk/actions/workflows/quality.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

</div>

---

Replace `import nltk` with `import fastnltk as nltk` and everything works.
Same [NLTK API](https://www.nltk.org/api/nltk.html), same [corpus data](https://www.nltk.org/data.html),
same results — but tokenization, tagging, parsing, stemming, and metrics run
in Rust instead of Python. No rewrites, no new API to learn.

## Install

```bash
pip install fastnltk
```

Pre-built wheels for Linux (x86_64, arm64), macOS (x86_64, arm64), Windows (x64).
Python 3.8–3.13. Uses your existing NLTK data — no re-downloading.

```bash
# If you don't have NLTK data yet:
python -m nltk.downloader punkt averaged_perceptron_tagger wordnet
```

## Why

NLTK is the most-used NLP teaching library. But it's pure Python — regex loops
and dict lookups run through the interpreter, 10–50× slower than compiled code.
Some operations got **worse** across NLTK versions (0.55s → 216s on 30K chars).

fastNLTK swaps the engine without changing the interface.

## Benchmarks

61 benchmarks. **Geometric mean: 7.6× faster** than NLTK on equivalent operations.
[Full results →](BENCHMARKS.md)

| Operation | NLTK | fastNLTK | Speedup |
|---|---:|---:|---:|
| windowdiff | 2.47 ms | 0.01 ms | **168×** |
| edit_distance | 2.44 ms | 0.02 ms | **152×** |
| pk | 2.20 ms | 0.02 ms | **93×** |
| TreebankWordDetokenizer | 6.96 ms | 0.20 ms | **35×** |
| PunktSentenceTokenizer | 14.56 ms | 0.43 ms | **34×** |
| Expression.fromstring | 17.45 ms | 0.58 ms | **30×** |
| PerceptronTagger | 17.64 ms | 0.66 ms | **27×** |
| CFG.from_string | 0.05 ms | 0.002 ms | **26×** |
| TweetTokenizer | 85.44 ms | 3.51 ms | **24×** |
| LancasterStemmer | 33.86 ms | 2.01 ms | **17×** |
| TreebankWordTokenizer | 42.56 ms | 2.54 ms | **17×** |

Best modules: metrics 133×, sem 30×, parse 24×, collocations 13×,
tree 10×, translate 8×, stem 8×, chunk 8×.

## Usage

```python
from fastnltk import word_tokenize, pos_tag, sent_tokenize
from fastnltk import Tree

# Tokenize
word_tokenize("Dr. Smith can't believe how fast this is.")
# ['Dr.', 'Smith', 'ca', "n't", 'believe', 'how', 'fast', 'this', 'is', '.']

# Sentence segmentation
sent_tokenize("Dr. Smith went home. He ate dinner.")
# ['Dr. Smith went home.', 'He ate dinner.']

# POS tagging
pos_tag("the quick brown fox".split())
# [('the', 'DT'), ('quick', 'JJ'), ('brown', 'NN'), ('fox', 'NN')]

# Parse trees
tree = Tree.from_string("(S (NP The/DT cat/NN) (VP runs/VBZ))")
tree.leaves()       # ['The/DT', 'cat/NN', 'runs/VBZ']
tree.productions()  # ['S -> NP VP', 'NP -> The/DT cat/NN', 'VP -> runs/VBZ']
```

Or go module-by-module — everything NLTK exposes is available:

```python
from fastnltk.stem import PorterStemmer, LancasterStemmer
from fastnltk.tag import PerceptronTagger, TnT
from fastnltk.probability import FreqDist, ConditionalFreqDist
from fastnltk.lm import MLE, Laplace, KneserNeyInterpolated
from fastnltk.metrics import edit_distance, jaro_winkler_similarity
from fastnltk.parse import CFG, EarleyChartParser
from fastnltk.classify import NaiveBayesClassifier, MaxentClassifier
from fastnltk.collocations import BigramCollocationFinder
from fastnltk.cluster import KMeansClusterer
from fastnltk.translate import bleu_score, IBMModel1
from fastnltk.chat import eliza_chat
```

## What's accelerated

Everything that touches text data runs in Rust. Pure-Python NLTK classes
(classifiers, parsers, corpus readers) pass through. See
[NLTK API docs](https://www.nltk.org/api/nltk.html) for full reference.

| Module | Rust-backed | Notes |
|---|---|---|
| tokenize | 14 tokenizers | Treebank, Toktok, Tweet, Punkt, Regexp, SExpr, MWE, TextTiling |
| stem | 9 stemmers | Porter, Lancaster, Snowball, WordNet, ISRI, RSLP, Cistem, ARLSTem |
| tag | 8 taggers | Perceptron, TnT, HMM, Ngram (1–3), Regexp, Affix |
| probability | FreqDist + 10 dists | MLE, Laplace, Lidstone, WittenBell, ELE, SGT, Uniform |
| lm | 6 models | MLE, Laplace, Lidstone, KneserNey, WittenBell, StupidBackoff |
| metrics | 14 functions | edit_distance, jaccard, jaro, windowdiff, pk, BLEU, AER |
| collocations | 3 finders | Bigram, Trigram, Quadgram + assoc measures |
| classify | 3 classifiers | NaiveBayes, Maxent, DecisionTree |
| parse | CFG + 6 parsers | Earley, Chart, BottomUp, LeftCorner, Stepping, ShiftReduce |
| tree | Tree + 5 variants | ParentedTree, ImmutableTree, ProbabilisticTree |
| chunk | Regexp chunker | NP/VP extraction, conll/tree conversion |
| sentiment | VADER | Same compound scores |
| sem | Expression parser | FOL parsing, model evaluation |
| cluster | KMeans, EM, GAAC | Euclidean, cosine distance |
| translate | BLEU + IBM1 | sentence_bleu, corpus_bleu |
| chat | 5 chatbots | Eliza, Iesha, Rude, Suntsu, Zen |
| ccg | CCG parser | Chart, combinators |
| inference | 3 provers | Tableau, Resolution, Discourse |

## Tests

- **Rust**: 312 unit tests — [crate-level `unsafe_code = "deny"`](https://doc.rust-lang.org/rustc/lints/listing/warn-by-default.html#unsafe-code)
- **Python**: 107 integration tests verify output matches NLTK byte-for-byte
- **CI**: `cargo clippy`, `cargo fmt --check`, `ruff`, cross-platform wheel builds

## Develop

```bash
git clone https://github.com/fastnltk/fastnltk
cd fastnltk
pip install -e ".[dev]"
maturin develop --release

cargo test       # 312 Rust tests
pytest tests/     # 107 Python tests

cargo fmt --all -- --check
cargo clippy --all-targets
ruff check fastnltk/ tests/

python -m benchmarks.run --save
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

Apache 2.0. See [LICENSE](LICENSE).

fastNLTK is not affiliated with NLTK or its maintainers.

## Cite

```bibtex
@software{fastnltk2025,
  author = {Wyatt Ferguson},
  title = {fastNLTK: Drop-in Rust-accelerated replacement for NLTK},
  year = {2025},
  url = {https://github.com/fastnltk/fastnltk}
}
```

---

Made by [Wyatt Ferguson](https://github.com/wyattferguson) ·
[wyattf.bsky.social](https://wyattf.bsky.social) ·
[wyattxdev@duck.com](mailto:wyattxdev@duck.com)
