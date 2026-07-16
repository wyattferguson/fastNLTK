# Migrating from NLTK to fastNLTK

fastNLTK is a drop-in replacement for NLTK. In most cases, changing your import is all you need:

```python
# Before
import nltk
tokens = nltk.word_tokenize("The quick brown fox.")

# After
import fastnltk as nltk
tokens = nltk.word_tokenize("The quick brown fox.")
```

All your NLTK data (corpora, models, pickles) still works. Nothing to re-download.

## Known behavioral differences

These edge cases differ from NLTK. If your code depends on exact NLTK-internal behavior, review this list.

### Punkt sentence tokenizer — quote-start detection

NLTK treats `"` + capital letter as a sentence boundary heuristic (e.g., `He said "Go."` splits after `said`). fastNLTK's Rust Punkt does not implement this heuristic. This affects only texts with quotation marks starting sentences.

### ConditionalFreqDist clone semantics

`ConditionalFreqDist.freqdist(cond)` returns a **clone** of the `FreqDist`. In NLTK, the returned object shares references with the parent — mutations propagate back. In fastNLTK, mutations to the returned `FreqDist` do **not** affect the parent.

Workaround: use `ConditionalFreqDist.inc(cond, sample)` to update in-place.

### Earley chart parser — tree extraction

The Rust Earley parser finds the same parses as NLTK's, but the tree structure produced by `parse()` may differ from NLTK's chart-printing format. For CFG parsing with `EarleyChartParser`, use `CFG.from_string()` for grammar parsing (identical output).

### BigramAssocMeasures — repr precision

NLTK 3.10's `student_t`/`chi_sq` internal computation has edge-case behavior in `repr()` formatting. The scoring values are identical; only the string representation differs.

### AffixTagger — untrained model

The Rust `AffixTagger` requires training data to infer the tagset. NLTK's Python version defaults to `None`. Call `.train(sentences)` before tagging.

### VADER — `but` handling

The Rust VADER implements NLTK's `_but_check` (words before "but" get ×0.5, after get ×1.5). However, the `_least_check`, `_idioms_check`, and `_never_check` with "never X so/this" special cases are simplified. For most texts the scores match to within 0.001.

### WordNet lemmatizer — data loading

fastNLTK loads WordNet data from `nltk_data/corpora/wordnet/`. If your NLTK data is in a `.zip` archive (common on Windows), the Python wrapper will extract it to a directory on first use. Ensure `nltk_data` is accessible via `NLTK_DATA`, `APPDATA`, `HOME`, or `USERPROFILE` environment variables.

### PerceptronTagger — model caching

The Rust `PerceptronTagger` caches model weights using a deterministic `FxHash` (not randomized like `rustc-hash` v2). Model files from NLTK's pickled format are compatible — fastNLTK loads them through Python during `tag()` and converts weights.

## API coverage

All NLTK module paths work under `fastnltk.*`:

```python
from fastnltk.tokenize import word_tokenize, sent_tokenize
from fastnltk.tag import PerceptronTagger, TnT
from fastnltk.stem import PorterStemmer, LancasterStemmer
from fastnltk.probability import FreqDist, ConditionalFreqDist
from fastnltk.collocations import BigramCollocationFinder
from fastnltk.metrics import edit_distance, jaccard_distance
from fastnltk.parse import CFG, EarleyChartParser
from fastnltk.tree import Tree
from fastnltk.sentiment import SentimentIntensityAnalyzer
```

## Performance notes

- **Geometric mean: 9.2× faster** across 49 comparable benchmarks
- **Tokenizers**: 10–697× faster (SIMD memchr3, single-pass char scanners)
- **POS taggers**: 3–8× faster (u64 feature IDs, integer Viterbi)
- **Stemmers**: 7–18× faster (pure Rust implementations)
- **VADER**: 40× faster (PHF lexicon, zero-allocation word scan)
- **edit_distance / windowdiff / pk**: 100–200× faster

## Reporting issues

If you find a case where fastNLTK's output differs from NLTK's beyond what's documented here, please open an issue at:
https://github.com/wyattferguson/fastnltk/issues
