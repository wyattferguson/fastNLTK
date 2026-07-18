"""
Benchmark suite — all benchmark definitions for fastNLTK.

Each function returns a `BenchResult` or list of `BenchResult`.

Covers all Rust modules: tokenize, stem, tag, classify, lm, probability,
collocations, ccg, inference, metrics, chunk, cluster, sem, translate,
sentiment, chat, tree, parse.
"""

import json

from .harness import (
    BenchResult,
    _median_time,
    fixture,
)

# ── Helpers ───────────────────────────────────────────────


def _ensure_data():
    """Download NLTK data if needed."""
    import nltk

    for resource in ["punkt", "averaged_perceptron_tagger", "wordnet", "vader_lexicon"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource)


# ── Tokenizers ────────────────────────────────────────────


def bench_toktok() -> BenchResult:
    import nltk.tokenize

    from fastnltk._rust import ToktokTokenizer

    text = fixture("medium")
    ntk = nltk.tokenize.ToktokTokenizer()
    rust = ToktokTokenizer()
    n_ms = _median_time(lambda: ntk.tokenize(text), 30)
    f_ms = _median_time(lambda: rust.tokenize(text, False), 30)
    return BenchResult(
        name="ToktokTokenizer.tokenize",
        group="tokenize",
        params={"chars": len(text)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=30,
    )


def bench_mwe() -> BenchResult:
    from nltk.tokenize import MWETokenizer as NltkMWE

    from fastnltk._rust import MWETokenizer as FastMWE

    words = fixture("medium").split()[:18000]
    ntk = NltkMWE([("New", "York")])
    rust = FastMWE([["New", "York"]], "_")
    n_ms = _median_time(lambda: ntk.tokenize(words), 30)
    f_ms = _median_time(lambda: rust.tokenize(words), 30)
    return BenchResult(
        name="MWETokenizer.tokenize",
        group="tokenize",
        params={"words": len(words)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=30,
    )


def bench_texttiling() -> BenchResult:
    from nltk.tokenize.texttiling import TextTilingTokenizer as NltkTT

    from fastnltk._rust import TextTilingTokenizer

    # Build text with paragraph breaks (needed by NLTK's TextTiling)
    paras = []
    for i in range(50):
        paras.append(fixture("tiny").strip() + "\n\n")
    text = "".join(paras)
    ntk = NltkTT()
    rust = TextTilingTokenizer(20, 10, True)
    n_ms = _median_time(lambda: ntk.tokenize(text), 5)
    f_ms = _median_time(lambda: rust.tokenize(text), 5)
    return BenchResult(
        name="TextTilingTokenizer.tokenize",
        group="tokenize",
        params={"chars": len(text)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=5,
    )


def bench_punkt_sent_tokenize() -> BenchResult:
    from nltk.tokenize.punkt import PunktSentenceTokenizer as NltkPunkt

    from fastnltk._rust import PunktSentenceTokenizer

    text = fixture("medium")
    ntk = NltkPunkt()
    rust = PunktSentenceTokenizer()
    n_ms = _median_time(lambda: ntk.tokenize(text), 15)
    f_ms = _median_time(lambda: rust.tokenize(text), 15)
    return BenchResult(
        name="PunktSentenceTokenizer.tokenize",
        group="tokenize",
        params={"chars": len(text)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=15,
    )


def bench_regexp_tokenizer() -> BenchResult:
    import nltk.tokenize

    from fastnltk._rust import RegexpTokenizer

    text = fixture("medium")
    ntk = nltk.tokenize.RegexpTokenizer(r"\w+")
    rust = RegexpTokenizer(r"\w+")
    n_ms = _median_time(lambda: ntk.tokenize(text), 30)
    f_ms = _median_time(lambda: rust.tokenize(text), 30)
    return BenchResult(
        name="RegexpTokenizer.tokenize",
        group="tokenize",
        params={"chars": len(text)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=30,
    )


def bench_treebank_tokenizer() -> BenchResult:
    import nltk.tokenize

    from fastnltk._rust import TreebankWordTokenizer

    text = fixture("medium")
    ntk = nltk.tokenize.TreebankWordTokenizer()
    rust = TreebankWordTokenizer()
    n_ms = _median_time(lambda: ntk.tokenize(text), 15)
    f_ms = _median_time(lambda: rust.tokenize(text), 15)
    return BenchResult(
        name="TreebankWordTokenizer.tokenize",
        group="tokenize",
        params={"chars": len(text)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=15,
    )


def bench_tweet_tokenizer() -> BenchResult:
    import nltk.tokenize

    from fastnltk._rust import TweetTokenizer

    text = fixture("medium")
    ntk = nltk.tokenize.TweetTokenizer()
    rust = TweetTokenizer()
    n_ms = _median_time(lambda: ntk.tokenize(text), 15)
    f_ms = _median_time(lambda: rust.tokenize(text), 15)
    return BenchResult(
        name="TweetTokenizer.tokenize",
        group="tokenize",
        params={"chars": len(text)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=15,
    )


def bench_sexpr_tokenizer() -> BenchResult:
    import nltk.tokenize

    from fastnltk._rust import SExprTokenizer

    text = "(a (b c)) (d (e f)) " * 200
    ntk = nltk.tokenize.SExprTokenizer()
    rust = SExprTokenizer("()", True)
    n_ms = _median_time(lambda: ntk.tokenize(text), 30)
    f_ms = _median_time(lambda: rust.tokenize(text), 30)
    return BenchResult(
        name="SExprTokenizer.tokenize",
        group="tokenize",
        params={"chars": len(text)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=30,
    )


def bench_detokenizer() -> BenchResult:
    import nltk.tokenize

    from fastnltk._rust import TreebankWordDetokenizer

    tokens = fixture("medium").split()[:5000]
    ntk = nltk.tokenize.TreebankWordDetokenizer()
    rust = TreebankWordDetokenizer()
    n_ms = _median_time(lambda: ntk.detokenize(tokens), 30)
    f_ms = _median_time(lambda: rust.detokenize(tokens), 30)
    return BenchResult(
        name="TreebankWordDetokenizer.detokenize",
        group="tokenize",
        params={"tokens": len(tokens)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=30,
    )


def bench_tab_tokenizer() -> BenchResult:
    import nltk.tokenize

    from fastnltk._rust import TabTokenizer

    text = fixture("medium")
    ntk = nltk.tokenize.TabTokenizer()
    rust = TabTokenizer()
    n_ms = _median_time(lambda: ntk.tokenize(text), 30)
    f_ms = _median_time(lambda: rust.tokenize(text), 30)
    return BenchResult(
        name="TabTokenizer.tokenize",
        group="tokenize",
        params={"chars": len(text)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=30,
    )


def bench_line_tokenizer() -> BenchResult:
    import nltk.tokenize

    from fastnltk._rust import LineTokenizer

    text = fixture("medium")
    ntk = nltk.tokenize.LineTokenizer()
    rust = LineTokenizer()
    n_ms = _median_time(lambda: ntk.tokenize(text), 30)
    f_ms = _median_time(lambda: rust.tokenize(text), 30)
    return BenchResult(
        name="LineTokenizer.tokenize",
        group="tokenize",
        params={"chars": len(text)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=30,
    )


def bench_whitespace_tokenizer() -> BenchResult:
    import nltk.tokenize

    from fastnltk._rust import WhitespaceTokenizer

    text = fixture("medium")
    ntk = nltk.tokenize.WhitespaceTokenizer()
    rust = WhitespaceTokenizer()
    n_ms = _median_time(lambda: ntk.tokenize(text), 30)
    f_ms = _median_time(lambda: rust.tokenize(text), 30)
    return BenchResult(
        name="WhitespaceTokenizer.tokenize",
        group="tokenize",
        params={"chars": len(text)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=30,
    )


def bench_wordpunct_tokenizer() -> BenchResult:
    import nltk.tokenize

    from fastnltk._rust import WordPunctTokenizer

    text = fixture("medium")
    ntk = nltk.tokenize.WordPunctTokenizer()
    rust = WordPunctTokenizer()
    n_ms = _median_time(lambda: ntk.tokenize(text), 30)
    f_ms = _median_time(lambda: rust.tokenize(text), 30)
    return BenchResult(
        name="WordPunctTokenizer.tokenize",
        group="tokenize",
        params={"chars": len(text)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=30,
    )


def bench_blankline_tokenizer() -> BenchResult:
    import nltk.tokenize

    from fastnltk._rust import BlanklineTokenizer

    text = fixture("medium")
    ntk = nltk.tokenize.BlanklineTokenizer()
    rust = BlanklineTokenizer()
    n_ms = _median_time(lambda: ntk.tokenize(text), 30)
    f_ms = _median_time(lambda: rust.tokenize(text), 30)
    return BenchResult(
        name="BlanklineTokenizer.tokenize",
        group="tokenize",
        params={"chars": len(text)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=30,
    )


def bench_space_tokenizer() -> BenchResult:
    import nltk.tokenize

    from fastnltk._rust import SpaceTokenizer

    text = fixture("medium")
    ntk = nltk.tokenize.SpaceTokenizer()
    rust = SpaceTokenizer()
    n_ms = _median_time(lambda: ntk.tokenize(text), 30)
    f_ms = _median_time(lambda: rust.tokenize(text), 30)
    return BenchResult(
        name="SpaceTokenizer.tokenize",
        group="tokenize",
        params={"chars": len(text)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=30,
    )


def bench_logos_tokenizer() -> BenchResult:
    from fastnltk._rust import logos_word_tokenize

    text = fixture("medium")
    f_ms = _median_time(lambda: logos_word_tokenize(text), 30)
    return BenchResult(
        name="logos_word_tokenize",
        group="tokenize",
        params={"chars": len(text)},
        fast_only_ms=f_ms,
        iterations=30,
    )


# ── Stemming ──────────────────────────────────────────────


def bench_snowball() -> BenchResult:
    import nltk.stem.snowball as nltk_sb

    from fastnltk._rust import SnowballStemmer

    words = (fixture("medium").split() * 3)[:10000]
    ntk = nltk_sb.SnowballStemmer("english")
    rust = SnowballStemmer("english")
    n_ms = _median_time(lambda: [ntk.stem(w) for w in words], 30)
    f_ms = _median_time(lambda: rust.stem_many(words), 30)
    return BenchResult(
        name="SnowballStemmer.stem",
        group="stem",
        params={"words": len(words)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=30,
    )


def bench_porter() -> BenchResult:
    import nltk.stem.porter as nltk_pt

    from fastnltk._rust import PorterStemmer

    words = (fixture("medium").split() * 3)[:10000]
    ntk = nltk_pt.PorterStemmer()
    rust = PorterStemmer()
    n_ms = _median_time(lambda: [ntk.stem(w) for w in words], 30)
    f_ms = _median_time(lambda: [rust.stem(w) for w in words], 30)
    return BenchResult(
        name="PorterStemmer.stem",
        group="stem",
        params={"words": len(words)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=30,
    )


def bench_lancaster() -> BenchResult:
    import nltk.stem.lancaster as nltk_lc

    from fastnltk._rust import LancasterStemmer

    words = (fixture("medium").split() * 3)[:10000]
    ntk = nltk_lc.LancasterStemmer()
    rust = LancasterStemmer()
    n_ms = _median_time(lambda: [ntk.stem(w) for w in words], 30)
    f_ms = _median_time(lambda: [rust.stem(w) for w in words], 30)
    return BenchResult(
        name="LancasterStemmer.stem",
        group="stem",
        params={"words": len(words)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=30,
    )


def bench_wordnet() -> BenchResult:
    from nltk.stem import WordNetLemmatizer as NltkWN

    from fastnltk._rust import WordNetLemmatizer

    words = (fixture("medium").split() * 3)[:5000]
    ntk = NltkWN()
    rust = WordNetLemmatizer()
    n_ms = _median_time(lambda: [ntk.lemmatize(w) for w in words], 15)
    f_ms = _median_time(lambda: [rust.lemmatize(w) for w in words], 15)
    return BenchResult(
        name="WordNetLemmatizer.lemmatize",
        group="stem",
        params={"words": len(words)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=15,
    )


def bench_arlstem() -> BenchResult:
    from nltk.stem.arlstem import ARLSTem as NltkARL

    from fastnltk._rust import ARLSTem

    words = ["كتب", "الكتاب", "مكتبة"] * 500
    ntk = NltkARL()
    rust = ARLSTem()
    n_ms = _median_time(lambda: [ntk.stem(w) for w in words], 30)
    f_ms = _median_time(lambda: [rust.stem(w) for w in words], 30)
    return BenchResult(
        name="ARLSTem.stem",
        group="stem",
        params={"words": len(words)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=30,
    )


def bench_isri_stemmer() -> BenchResult:
    from nltk.stem.isri import ISRIStemmer as NltkISRI

    from fastnltk._rust import ISRIStemmer

    words = ["كتب", "الكتاب", "مكتبة"] * 500
    ntk = NltkISRI()
    rust = ISRIStemmer()
    n_ms = _median_time(lambda: [ntk.stem(w) for w in words], 30)
    f_ms = _median_time(lambda: [rust.stem(w) for w in words], 30)
    return BenchResult(
        name="ISRIStemmer.stem",
        group="stem",
        params={"words": len(words)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=30,
    )


def bench_rslp_stemmer() -> BenchResult:
    from fastnltk._rust import RSLPStemmer

    # NLTK RSLPStemmer requires nltk.download('rslp') data — skip comparison
    words = ["correndo", "correr", "correu", "corria"] * 500
    rust = RSLPStemmer()
    f_ms = _median_time(lambda: [rust.stem(w) for w in words], 50)
    return BenchResult(
        name="RSLPStemmer.stem",
        group="stem",
        params={"words": len(words)},
        fast_only_ms=f_ms,
        iterations=50,
    )


def bench_regexp_stemmer() -> BenchResult:
    from fastnltk._rust import RegexpStemmer

    # Rust RegexpStemmer uses hardcoded patterns; NLTK version takes custom regex
    words = ["running", "jumping", "walking", "eating", "happily", "kindness"] * 500
    rust = RegexpStemmer()
    f_ms = _median_time(lambda: [rust.stem(w) for w in words], 50)
    return BenchResult(
        name="RegexpStemmer.stem",
        group="stem",
        params={"words": len(words)},
        fast_only_ms=f_ms,
        iterations=50,
    )


# ── POS Tagging ──────────────────────────────────────────


def bench_perceptron_tagger() -> BenchResult:
    import nltk.tag

    from fastnltk.tag import pos_tag

    words = fixture("medium").split()[:3000]
    sents = [words[i : i + 10] for i in range(0, min(len(words), 1000), 10)][:100]

    ntk = nltk.tag.PerceptronTagger()
    _ = [ntk.tag(s) for s in sents[:5]]  # warmup
    n_ms = _median_time(lambda: [ntk.tag(s) for s in sents], 5)
    f_ms = _median_time(lambda: [pos_tag(s) for s in sents], 5)
    return BenchResult(
        name="PerceptronTagger.tag",
        group="tag",
        params={"sentences": len(sents)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=5,
    )


def bench_hmm_tag() -> BenchResult:
    import nltk.tag.hmm

    from fastnltk._rust import HiddenMarkovModelTagger

    train = [
        [("I", "PRP"), ("like", "VBP"), ("dogs", "NNS")],
        [("She", "PRP"), ("runs", "VBZ"), ("fast", "RB")],
    ]
    words_1k = ["I", "like", "dogs"] * 333

    # NLTK HMM
    ntk = nltk.tag.hmm.HiddenMarkovModelTagger.train([[(w, t) for w, t in sent] for sent in train])
    n_ms = _median_time(lambda: ntk.tag(words_1k), 15)

    # fastNLTK HMM
    hm = HiddenMarkovModelTagger()
    hm.train(train)
    f_ms = _median_time(lambda: hm.tag(words_1k), 15)

    return BenchResult(
        name="HiddenMarkovModelTagger.tag",
        group="tag",
        params={"words": len(words_1k)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=15,
    )


def bench_tnt_tag() -> BenchResult:
    import nltk.tag

    from fastnltk._rust import TnT

    train = [
        [("the", "DT"), ("cat", "NN")],
        [("the", "DT"), ("dog", "NN")],
        [("a", "DT"), ("fox", "NN")],
        [("a", "DT"), ("bear", "NN")],
    ] * 5
    # NLTK TnT
    ntk = nltk.tag.TnT()
    ntk.train(train)
    words = ["the", "cat", "a", "dog", "the", "fox"] * 166
    n_ms = _median_time(lambda: ntk.tag(words), 15)
    # fastNLTK TnT
    rust = TnT()
    rust.train(train)
    f_ms = _median_time(lambda: rust.tag(words), 15)
    return BenchResult(
        name="TnT.tag",
        group="tag",
        params={"words": len(words)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=15,
    )


def bench_default_tagger() -> BenchResult:
    import nltk.tag

    from fastnltk._rust import DefaultTagger

    words = fixture("medium").split()[:10000]
    ntk = nltk.tag.DefaultTagger("NN")
    rust = DefaultTagger("NN")
    n_ms = _median_time(lambda: ntk.tag(words), 30)
    f_ms = _median_time(lambda: rust.tag(words), 30)
    return BenchResult(
        name="DefaultTagger.tag",
        group="tag",
        params={"words": len(words)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=30,
    )


# ── Sequential Taggers ────────────────────────────────────


def _train_data_words(py):
    from pyo3 import Python

    train = [
        [("the", "DT"), ("cat", "NN")],
        [("the", "DT"), ("dog", "NN")],
        [("a", "DT"), ("fox", "NN")],
    ]
    return train


def bench_unigram_tagger() -> BenchResult:
    import nltk.tag

    from fastnltk._rust import UnigramTagger

    train = [[("the", "DT"), ("cat", "NN")], [("the", "DT"), ("dog", "NN")]]
    words = ["the", "cat", "dog", "fox", "run"] * 2000

    ntk = nltk.tag.UnigramTagger(train)
    rust = UnigramTagger()
    rust.train(train)

    n_ms = _median_time(lambda: ntk.tag(words), 30)
    f_ms = _median_time(lambda: rust.tag(words), 30)
    return BenchResult(
        name="UnigramTagger.tag",
        group="tag",
        params={"words": len(words)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=30,
    )


def bench_bigram_tagger() -> BenchResult:
    import nltk.tag

    from fastnltk._rust import BigramTagger

    train = [[("the", "DT"), ("cat", "NN")], [("the", "DT"), ("dog", "NN")]]
    words = ["the", "cat", "the", "dog"] * 2500

    ntk = nltk.tag.BigramTagger(train)
    rust = BigramTagger()
    rust.train(train)

    n_ms = _median_time(lambda: ntk.tag(words), 30)
    f_ms = _median_time(lambda: rust.tag(words), 30)
    return BenchResult(
        name="BigramTagger.tag",
        group="tag",
        params={"words": len(words)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=30,
    )


def bench_trigram_tagger() -> BenchResult:
    import nltk.tag

    from fastnltk._rust import TrigramTagger

    train = [
        [("the", "DT"), ("cat", "NN"), ("runs", "VBZ")],
        [("the", "DT"), ("dog", "NN"), ("sleeps", "VBZ")],
    ]
    words = ["the", "cat", "runs", "the", "dog", "sleeps"] * 1700

    ntk = nltk.tag.TrigramTagger(train)
    rust = TrigramTagger()
    rust.train(train)

    n_ms = _median_time(lambda: ntk.tag(words), 30)
    f_ms = _median_time(lambda: rust.tag(words), 30)
    return BenchResult(
        name="TrigramTagger.tag",
        group="tag",
        params={"words": len(words)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=30,
    )


def bench_regexp_tagger() -> BenchResult:
    import nltk.tag

    from fastnltk._rust import RegexpTagger

    patterns = [(r"\d+", "CD"), (r"[A-Z].*", "NNP")]
    words = ["123", "John", "hello", "world", "42"] * 2000

    ntk = nltk.tag.RegexpTagger(patterns)
    rust = RegexpTagger(patterns, None)

    n_ms = _median_time(lambda: ntk.tag(words), 30)
    f_ms = _median_time(lambda: rust.tag(words), 30)
    return BenchResult(
        name="RegexpTagger.tag",
        group="tag",
        params={"words": len(words)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=30,
    )


def bench_affix_tagger() -> BenchResult:
    import nltk.tag

    from fastnltk._rust import AffixTagger

    train = [[("walking", "VBG"), ("running", "VBG"), ("eats", "VBZ")]]
    words = ["walking", "running", "eats", "talks"] * 2500

    ntk = nltk.tag.AffixTagger(train, affix_length=3, min_stem_length=1)
    rust = AffixTagger(3, True, None)
    rust.train(train)

    n_ms = _median_time(lambda: ntk.tag(words), 30)
    f_ms = _median_time(lambda: rust.tag(words), 30)
    return BenchResult(
        name="AffixTagger.tag",
        group="tag",
        params={"words": len(words)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=30,
    )


# ── Classification ────────────────────────────────────────


def bench_naivebayes_train() -> BenchResult:
    from nltk.classify import NaiveBayesClassifier as NltkNB

    from fastnltk._rust import NaiveBayesClassifier as FastNB

    # Build 2000 training instances
    train_data_list = []
    for i in range(2000):
        label = "pos" if i % 2 == 0 else "neg"
        feats = {f"feat_{j}": str((i + j) % 4) for j in range(10)}
        train_data_list.append((feats, label))

    def run_nltk():
        NltkNB.train(train_data_list)

    def run_fast():
        nb = FastNB()
        nb.train(train_data_list, 1.0)

    n_ms = _median_time(run_nltk, 5)
    f_ms = _median_time(run_fast, 10)
    return BenchResult(
        name="NaiveBayesClassifier.train",
        group="classify",
        params={"instances": len(train_data_list)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=10,
    )


def bench_naivebayes_classify() -> BenchResult:
    from nltk.classify import NaiveBayesClassifier as NltkNB

    from fastnltk._rust import NaiveBayesClassifier as FastNB

    train_data = []
    for i in range(500):
        label = "pos" if i % 2 == 0 else "neg"
        feats = {f"feat_{j}": str((i + j) % 3) for j in range(5)}
        train_data.append((feats, label))

    ntk = NltkNB.train(train_data)
    nb = FastNB()
    nb.train(train_data, 1.0)

    test_feats = {f"feat_{j}": str(j % 3) for j in range(5)}
    n_ms = _median_time(lambda: ntk.classify(test_feats), 100)
    f_ms = _median_time(lambda: nb.classify(test_feats), 100)
    return BenchResult(
        name="NaiveBayesClassifier.classify",
        group="classify",
        params={"features": len(test_feats)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=100,
    )


def bench_maxent_train() -> BenchResult:
    from nltk.classify import MaxentClassifier as NltkMaxent

    from fastnltk._rust import MaxentClassifier

    train_data = []
    for i in range(200):
        label = "pos" if i % 2 == 0 else "neg"
        feats = {f"feat_{j}": str((i + j) % 3) for j in range(5)}
        train_data.append((feats, label))

    n_ms = _median_time(lambda: NltkMaxent.train(train_data, max_iter=10, trace=0), 3)
    rust = MaxentClassifier()
    f_ms = _median_time(lambda: rust.train(train_data, 10, 0.0), 3)
    return BenchResult(
        name="MaxentClassifier.train",
        group="classify",
        params={"instances": len(train_data)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=3,
    )


def bench_textcat() -> BenchResult:
    from fastnltk._rust import TextCat

    texts = [
        "the quick brown fox jumps over the lazy dog",
        "der schnelle braune Fuchs springt uber den faulen Hund",
        "le rapide renard brun saute par-dessus le chien paresseux",
        "el rapido zorro marron salta sobre el perro perezoso",
    ] * 25
    rust = TextCat()
    f_ms = _median_time(lambda: [rust.guess_language(t) for t in texts], 50)
    return BenchResult(
        name="TextCat.guess_language",
        group="classify",
        params={"texts": len(texts)},
        fast_only_ms=f_ms,
        iterations=50,
    )


# ── Probability ───────────────────────────────────────────


def bench_freqdist() -> BenchResult:
    import nltk.probability as nltk_prob

    from fastnltk._rust import FreqDist

    samples = fixture("medium").split() * 5
    samples = samples[:100000]

    def run_nltk():
        fd = nltk_prob.FreqDist()
        for s in samples:
            fd[s] += 1

    def run_fast():
        fd = FreqDist()
        fd.update(samples)

    n_ms = _median_time(run_nltk, 15)
    f_ms = _median_time(run_fast, 15)
    return BenchResult(
        name="FreqDist.update",
        group="probability",
        params={"samples": len(samples)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=15,
    )


def bench_conditional_freqdist() -> BenchResult:
    import nltk.probability as nltk_prob

    from fastnltk._rust import ConditionalFreqDist

    samples = fixture("medium").split()[:20000]
    conditions = [s[0] if s else "_" for s in samples]

    def run_nltk():
        cfd = nltk_prob.ConditionalFreqDist()
        for cond, samp in zip(conditions, samples):
            cfd[cond][samp] += 1

    def run_fast():
        cfd = ConditionalFreqDist()
        for cond, samp in zip(conditions, samples):
            cfd.inc(cond, samp)

    n_ms = _median_time(run_nltk, 15)
    f_ms = _median_time(run_fast, 15)
    return BenchResult(
        name="ConditionalFreqDist.inc",
        group="probability",
        params={"samples": len(samples)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=15,
    )


# ── Collocations ──────────────────────────────────────────


def bench_bigram_collocations() -> BenchResult:
    import nltk.collocations as nltk_coll

    from fastnltk._rust import BigramCollocationFinder

    words = (fixture("medium").split() * 5)[:50000]

    def run_nltk():
        finder = nltk_coll.BigramCollocationFinder.from_words(words, 2)
        finder.nbest(nltk_coll.BigramAssocMeasures().raw_freq, 10)

    def run_fast():
        finder = BigramCollocationFinder.from_words(words, 2)
        finder.nbest("raw_freq", 10)

    n_ms = _median_time(run_nltk, 15)
    f_ms = _median_time(run_fast, 15)
    return BenchResult(
        name="BigramCollocationFinder.from_words",
        group="collocations",
        params={"words": len(words)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=15,
    )


def bench_trigram_collocations() -> BenchResult:
    import nltk.collocations as nltk_coll

    from fastnltk._rust import TrigramCollocationFinder

    words = (fixture("medium").split() * 2)[:20000]

    def run_nltk():
        finder = nltk_coll.TrigramCollocationFinder.from_words(words)
        finder.nbest(nltk_coll.TrigramAssocMeasures().raw_freq, 5)

    def run_fast():
        finder = TrigramCollocationFinder.from_words(words, 3)
        finder.nbest("raw_freq", 5)

    n_ms = _median_time(run_nltk, 5)
    f_ms = _median_time(run_fast, 5)
    return BenchResult(
        name="TrigramCollocationFinder.from_words",
        group="collocations",
        params={"words": len(words)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=5,
    )


def bench_quadgram_collocations() -> BenchResult:
    import nltk.collocations as nltk_coll

    from fastnltk._rust import QuadgramCollocationFinder

    words = (fixture("medium").split() * 2)[:20000]

    def run_nltk():
        finder = nltk_coll.QuadgramCollocationFinder.from_words(words)
        finder.nbest(nltk_coll.QuadgramAssocMeasures().raw_freq, 5)

    def run_fast():
        finder = QuadgramCollocationFinder.from_words(words, 4)
        finder.nbest("raw_freq", 5)

    n_ms = _median_time(run_nltk, 5)
    f_ms = _median_time(run_fast, 5)
    return BenchResult(
        name="QuadgramCollocationFinder.from_words",
        group="collocations",
        params={"words": len(words)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=5,
    )


# ── Sentiment ─────────────────────────────────────────────


def bench_sentiment() -> BenchResult:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer as NltkVader

    from fastnltk._rust import SentimentIntensityAnalyzer

    text = fixture("medium")
    ntk = NltkVader()
    rust = SentimentIntensityAnalyzer()
    n_ms = _median_time(lambda: ntk.polarity_scores(text), 30)
    f_ms = _median_time(lambda: rust.polarity_scores(text), 50)
    return BenchResult(
        name="SentimentIntensityAnalyzer.polarity_scores",
        group="sentiment",
        params={"chars": len(text)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=50,
    )


# ── Segmentation Metrics ──────────────────────────────────


def bench_windowdiff() -> BenchResult:
    from nltk.metrics.segmentation import windowdiff as nwd

    from fastnltk._rust import windowdiff as fwd

    s1 = "000100000010" * 1000
    s2 = "000010000100" * 1000
    n_ms = _median_time(lambda: nwd(s1, s2, 3, "1", False), 100)
    f_ms = _median_time(lambda: fwd(s1, s2, 3, "1"), 100)
    return BenchResult(
        name="windowdiff",
        group="metrics",
        params={"chars": len(s1)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=100,
    )


def bench_pk() -> BenchResult:
    from nltk.metrics.segmentation import pk as npk

    from fastnltk._rust import pk as fpk

    s1 = "000100000010" * 1000
    s2 = "000010000100" * 1000
    n_ms = _median_time(lambda: npk(s1, s2, 3, "1"), 100)
    f_ms = _median_time(lambda: fpk(s1, s2, 3, "1"), 100)
    return BenchResult(
        name="pk",
        group="metrics",
        params={"chars": len(s1)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=100,
    )


def bench_edit_distance() -> BenchResult:
    from nltk.metrics.distance import edit_distance as nltk_ed

    from fastnltk._rust import edit_distance

    a = "abcdefghij" * 10
    b = "abxdefghij" * 10

    n_ms = _median_time(lambda: nltk_ed(a, b), 100)
    f_ms = _median_time(lambda: edit_distance(a, b), 100)
    return BenchResult(
        name="edit_distance",
        group="metrics",
        params={"chars": len(a)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=100,
    )


def bench_bigram_assoc_measures() -> BenchResult:
    from fastnltk._rust import BigramAssocMeasures

    args = (10.0, 1000.0, 8.0, 20.0, 30.0)
    f_ms = _median_time(
        lambda: [
            BigramAssocMeasures.pmi(*args),
            BigramAssocMeasures.chi_sq(*args),
            BigramAssocMeasures.likelihood_ratio(*args),
        ],
        500,
    )
    return BenchResult(
        name="BigramAssocMeasures",
        group="metrics",
        params={"calls": 3},
        fast_only_ms=f_ms,
        iterations=500,
    )


# ── LM ─────────────────────────────────────────────────────


def bench_mle_fit() -> BenchResult:
    from fastnltk.lm import MLE

    sentences = [
        ["the", "cat", "sat"],
        ["the", "dog", "ran"],
        ["a", "cat", "sleeps"],
        ["the", "mouse", "runs"],
    ] * 250
    m = MLE(2)
    m.fit(sentences)
    words = ["cat", "dog", "mouse", "rat"] * 250
    f_ms = _median_time(lambda: [m.score(w, ["the"]) for w in words], 30)
    return BenchResult(
        name="MLE.score",
        group="lm",
        params={"queries": len(words)},
        fast_only_ms=f_ms,
        iterations=30,
    )


def bench_lidstone() -> BenchResult:
    from fastnltk._rust import Lidstone

    sentences = [
        ["the", "cat", "sat"],
        ["the", "dog", "ran"],
        ["a", "cat", "sleeps"],
        ["the", "mouse", "runs"],
    ] * 250
    m = Lidstone(2, 0.2)
    m.fit([list(s) for s in sentences])
    words = ["cat", "dog", "mouse", "rat"] * 250
    f_ms = _median_time(lambda: [m.score(w, ["the"]) for w in words], 30)
    return BenchResult(
        name="Lidstone.score",
        group="lm",
        params={"queries": len(words)},
        fast_only_ms=f_ms,
        iterations=30,
    )


def bench_laplace_lm() -> BenchResult:
    from fastnltk._rust import Laplace

    sentences = [
        ["the", "cat", "sat"],
        ["the", "dog", "ran"],
        ["a", "cat", "sleeps"],
        ["the", "mouse", "runs"],
    ] * 250
    m = Laplace(2)
    m.fit([list(s) for s in sentences])
    words = ["cat", "dog", "mouse", "rat"] * 250
    f_ms = _median_time(lambda: [m.score(w, ["the"]) for w in words], 30)
    return BenchResult(
        name="Laplace.score",
        group="lm",
        params={"queries": len(words)},
        fast_only_ms=f_ms,
        iterations=30,
    )


def bench_stupid_backoff() -> BenchResult:
    from fastnltk._rust import StupidBackoff

    sentences = [
        ["the", "cat", "sat"],
        ["the", "dog", "ran"],
        ["a", "cat", "sleeps"],
        ["the", "mouse", "runs"],
    ] * 250
    m = StupidBackoff(2, 0.4)
    m.fit([list(s) for s in sentences])
    words = ["cat", "dog", "mouse", "rat"] * 250
    f_ms = _median_time(lambda: [m.score(w, ["the"]) for w in words], 30)
    return BenchResult(
        name="StupidBackoff.score",
        group="lm",
        params={"queries": len(words)},
        fast_only_ms=f_ms,
        iterations=30,
    )


def bench_kneser_ney() -> BenchResult:
    from fastnltk.lm import KneserNeyInterpolated

    sentences = [
        ["the", "cat", "sat"],
        ["the", "dog", "ran"],
        ["a", "cat", "sleeps"],
        ["the", "mouse", "runs"],
    ] * 250
    m = KneserNeyInterpolated(2, 0.75)
    m.fit([list(s) for s in sentences])
    words = ["cat", "dog", "mouse", "rat"] * 250
    f_ms = _median_time(lambda: [m.score(w, ["the"]) for w in words], 30)
    return BenchResult(
        name="KneserNeyInterpolated.score",
        group="lm",
        params={"queries": len(words)},
        fast_only_ms=f_ms,
        iterations=30,
    )


def bench_witten_bell() -> BenchResult:
    from fastnltk.lm import WittenBellInterpolated

    sentences = [
        ["the", "cat", "sat"],
        ["the", "dog", "ran"],
        ["a", "cat", "sleeps"],
        ["the", "mouse", "runs"],
    ] * 250
    m = WittenBellInterpolated(2)
    m.fit([list(s) for s in sentences])
    words = ["cat", "dog", "mouse", "rat"] * 250
    f_ms = _median_time(lambda: [m.score(w, ["the"]) for w in words], 30)
    return BenchResult(
        name="WittenBellInterpolated.score",
        group="lm",
        params={"queries": len(words)},
        fast_only_ms=f_ms,
        iterations=30,
    )


# ── Probability Distributions ──────────────────────────────


def bench_laplace_probdist() -> BenchResult:
    from fastnltk._rust import LaplaceProbDist, FreqDist

    fd = FreqDist()
    samples = ["a", "b", "c", "a", "b", "a"] * 500
    fd.update(samples)
    dist = LaplaceProbDist(fd)
    f_ms = _median_time(lambda: [dist.prob(w) for w in ["a", "b", "c", "d"]], 500)
    return BenchResult(
        name="LaplaceProbDist.prob",
        group="probability",
        params={"bins": 3},
        fast_only_ms=f_ms,
        iterations=500,
    )


def bench_mle_probdist() -> BenchResult:
    from fastnltk._rust import MLEProbDist, FreqDist

    fd = FreqDist()
    samples = ["a", "b", "c", "a", "b", "a"] * 500
    fd.update(samples)
    dist = MLEProbDist(fd)
    f_ms = _median_time(lambda: [dist.prob(w) for w in ["a", "b", "c", "d"]], 500)
    return BenchResult(
        name="MLEProbDist.prob",
        group="probability",
        params={"bins": 3},
        fast_only_ms=f_ms,
        iterations=500,
    )


# ── CCG ────────────────────────────────────────────────────


def bench_ccg_parse() -> BenchResult:
    from nltk.ccg.api import FunctionalCategory as NltkCat

    from fastnltk._rust import from_string as FastCat

    bs = chr(92)
    cats = [f"S{bs}NP", f"(S{bs}NP)/NP", "NP/N", "NP", "N", "PP", "S"] * 500

    def run_nltk():
        for c in cats:
            try:
                NltkCat.fromstring(c)
            except Exception:
                pass

    def run_fast():
        for c in cats:
            FastCat(c)

    n_ms = _median_time(run_nltk, 30)
    f_ms = _median_time(run_fast, 30)
    return BenchResult(
        name="CCG from_string",
        group="ccg",
        params={"parses": len(cats)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=30,
    )


# ── Chunking ──────────────────────────────────────────────


def bench_chunk_parse() -> BenchResult:
    import nltk.chunk as nltk_chunk

    from fastnltk._rust import RegexpParser

    grammar = "NP: {<DT>?<JJ>*<NN>}"
    ntk = nltk_chunk.RegexpParser(grammar)
    rust = RegexpParser(grammar)

    tokens = [
        ("The", "DT"),
        ("quick", "JJ"),
        ("brown", "JJ"),
        ("fox", "NN"),
        ("jumps", "VBZ"),
        ("over", "IN"),
        ("the", "DT"),
        ("lazy", "JJ"),
        ("dog", "NN"),
    ] * 200

    n_ms = _median_time(lambda: ntk.parse(tokens), 30)
    f_ms = _median_time(lambda: rust.parse(tokens), 30)
    return BenchResult(
        name="RegexpParser.parse",
        group="chunk",
        params={"tokens": len(tokens)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=30,
    )


# ── Clustering ────────────────────────────────────────────


def bench_kmeans() -> BenchResult:
    import numpy as np
    from nltk.cluster import KMeansClusterer as NltkKMeans
    from nltk.cluster.util import euclidean_distance

    from fastnltk._rust import KMeansClusterer

    vectors = [[float(i + j) for j in range(5)] for i in range(500)]
    # NLTK KMeans needs numpy arrays
    ntk_vectors = [np.array(v) for v in vectors]
    ntk = NltkKMeans(3, euclidean_distance, repeats=1)
    rust = KMeansClusterer(3, 100)

    def run_nltk():
        ntk.cluster(ntk_vectors, False)

    def run_fast():
        rust.cluster(vectors)

    n_ms = _median_time(run_nltk, 5)
    f_ms = _median_time(run_fast, 15)
    return BenchResult(
        name="KMeansClusterer.cluster",
        group="cluster",
        params={"vectors": len(vectors)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=15,
    )


# ── Parsing ───────────────────────────────────────────────


def bench_earley() -> BenchResult:
    import nltk
    from nltk.parse import EarleyChartParser as NltkEarley

    from fastnltk.parse import CFG, EarleyChartParser

    grammar_str = """S -> NP VP
NP -> Det N
NP -> N
VP -> V NP
Det -> 'the'
Det -> 'a'
N -> 'cat'
N -> 'dog'
N -> 'fox'
V -> 'chases'
V -> 'sees'"""
    ntk_grammar = nltk.CFG.fromstring(grammar_str)

    rust_grammar = CFG.from_string("""S -> NP VP
NP -> Det N
NP -> N
VP -> V NP
Det -> the
Det -> a
N -> cat
N -> dog
N -> fox
V -> chases
V -> sees""")

    sentences = [
        ["the", "cat", "chases", "the", "dog"],
        ["a", "dog", "sees", "the", "fox"],
        ["the", "fox", "chases", "a", "cat"],
    ] * 10

    ntk_parser = NltkEarley(ntk_grammar)
    rust_parser = EarleyChartParser()

    n_ms = _median_time(lambda: [list(ntk_parser.parse(s)) for s in sentences], 10)
    f_ms = _median_time(lambda: [rust_parser.parse(rust_grammar, s) for s in sentences], 15)
    return BenchResult(
        name="EarleyChartParser.parse",
        group="parse",
        params={"sentences": len(sentences)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=15,
    )


def bench_cfg() -> BenchResult:
    import nltk

    from fastnltk._rust import CFG

    grammar_str = """S -> NP VP
NP -> Det N | N
VP -> V NP | V
Det -> 'the' | 'a'
N -> 'cat' | 'dog' | 'fox'
V -> 'chases' | 'sees'"""
    n_ms = _median_time(lambda: nltk.CFG.fromstring(grammar_str), 100)
    f_ms = _median_time(lambda: CFG.from_string(grammar_str), 100)
    return BenchResult(
        name="CFG.from_string",
        group="parse",
        params={"rules": 9},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=100,
    )


# ── Translation ───────────────────────────────────────────


def bench_bleu() -> BenchResult:
    import nltk.translate.bleu_score as nltk_bleu

    from fastnltk._rust import bleu_score

    candidate = "the cat sat on the mat".split()
    reference = "the cat is on the mat".split()

    def run_nltk():
        nltk_bleu.sentence_bleu([reference], candidate)

    def run_fast():
        bleu_score(candidate, reference)

    n_ms = _median_time(run_nltk, 100)
    f_ms = _median_time(run_fast, 100)
    return BenchResult(
        name="bleu",
        group="translate",
        params={"tokens": len(candidate)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=100,
    )


# ── Chat ──────────────────────────────────────────────────


def bench_chat_respond() -> BenchResult:
    import nltk.chat.util as nltk_chat

    from fastnltk._rust import Chat

    pairs = [
        (r"hello|hi|hey", ["Hello!", "Hi there!"]),
        (r"how are you", ["I'm good, thanks.", "Doing well!"]),
        (r"what is your name", ["I'm a chatbot.", "Call me Bot."]),
    ]
    ntk = nltk_chat.Chat(pairs)
    rust = Chat(pairs)

    def run_nltk():
        ntk.respond("hello there")

    def run_fast():
        rust.respond("hello there")

    n_ms = _median_time(run_nltk, 100)
    f_ms = _median_time(run_fast, 100)
    return BenchResult(
        name="Chat.respond",
        group="chat",
        params={},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=100,
    )


# ── Tree ──────────────────────────────────────────────────


def bench_tree_from_string() -> BenchResult:
    import nltk.tree as nltk_tree

    from fastnltk._rust import Tree

    trees = [
        "(S (NP (Det the) (N cat)) (VP (V chases) (NP (Det the) (N dog))))",
        "(S (NP (N John)) (VP (V likes) (NP (Det a) (N book))))",
        "(S (NP (Det The) (JJ quick) (JJ brown) (N fox)) (VP (V jumps)))",
    ] * 100

    def run_nltk():
        for t in trees:
            nltk_tree.Tree.fromstring(t)

    def run_fast():
        for t in trees:
            Tree.from_string(t)

    n_ms = _median_time(run_nltk, 30)
    f_ms = _median_time(run_fast, 30)
    return BenchResult(
        name="Tree.from_string",
        group="tree",
        params={"trees": len(trees)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=30,
    )


# ── Semantics ─────────────────────────────────────────────


def bench_sem_expr() -> BenchResult:
    import nltk.sem.logic as nltk_sem

    from fastnltk._rust import fromstring

    formulas = [
        "exists x.(dog(x) & brown(x))",
        "all x.(man(x) -> mortal(x))",
        r"exists x.(dog(x) & all y.(cat(y) -> chase(x,y)))",
        r"\x.(dog(x) & brown(x))",
        r"(\x.(dog(x)))(fido)",
    ] * 100

    def run_nltk():
        for f in formulas:
            nltk_sem.Expression.fromstring(f)

    def run_fast():
        for f in formulas:
            fromstring(f)

    n_ms = _median_time(run_nltk, 30)
    f_ms = _median_time(run_fast, 30)
    return BenchResult(
        name="Expression.fromstring",
        group="sem",
        params={"formulas": len(formulas)},
        nltk_ms=n_ms,
        fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0,
        iterations=30,
    )


# ── Inference ─────────────────────────────────────────────


def bench_tableau() -> BenchResult:
    from fastnltk._rust import TableauProver

    tp = TableauProver(200)
    f_ms = _median_time(lambda: tp.prove("P | ~P", None), 50)
    return BenchResult(
        name="TableauProver.prove",
        group="inference",
        params={"formula": "P | ~P"},
        fast_only_ms=f_ms,
        iterations=50,
    )


def bench_resolution() -> BenchResult:
    from fastnltk._rust import ResolutionProver

    rp = ResolutionProver(1000)
    f_ms = _median_time(lambda: rp.prove("P | ~P", None), 50)
    return BenchResult(
        name="ResolutionProver.prove",
        group="inference",
        params={"formula": "P | ~P"},
        fast_only_ms=f_ms,
        iterations=50,
    )


def bench_discourse() -> BenchResult:
    from fastnltk._rust import DiscourseThread

    dt = DiscourseThread()
    dt.add_drs("([x],[dog(x)])")
    dt.add_drs("([y],[cat(y)])")
    val = json.dumps({"dog": [["fido"]], "cat": [["felix"]]})
    dom = json.dumps(["fido", "felix"])
    f_ms = _median_time(lambda: dt.answer_question("([x],[dog(x)])", val, dom), 50)
    return BenchResult(
        name="DiscourseThread.answer_question",
        group="inference",
        params={},
        fast_only_ms=f_ms,
        iterations=50,
    )


def bench_nonmonotonic() -> BenchResult:
    from fastnltk._rust import DefaultReasoner, DefaultRule

    rules = [DefaultRule("", f"fact{i}", f"fact{i}", "") for i in range(10)]
    r = DefaultReasoner(rules, 10)
    f_ms = _median_time(lambda: r.extensions(), 50)
    return BenchResult(
        name="DefaultReasoner.extensions",
        group="inference",
        params={"rules": 10},
        fast_only_ms=f_ms,
        iterations=50,
    )


# ── Registry ──────────────────────────────────────────────

ALL_BENCHMARKS: list[tuple[str, str, callable]] = [
    # tokenize (16)
    ("tokenize", "ToktokTokenizer", bench_toktok),
    ("tokenize", "MWETokenizer", bench_mwe),
    ("tokenize", "RegexpTokenizer", bench_regexp_tokenizer),
    ("tokenize", "SpaceTokenizer", bench_space_tokenizer),
    ("tokenize", "TreebankWordTokenizer", bench_treebank_tokenizer),
    ("tokenize", "TweetTokenizer", bench_tweet_tokenizer),
    ("tokenize", "TextTilingTokenizer", bench_texttiling),
    ("tokenize", "SExprTokenizer", bench_sexpr_tokenizer),
    ("tokenize", "PunktSentenceTokenizer", bench_punkt_sent_tokenize),
    ("tokenize", "TreebankWordDetokenizer", bench_detokenizer),
    ("tokenize", "TabTokenizer", bench_tab_tokenizer),
    ("tokenize", "LineTokenizer", bench_line_tokenizer),
    ("tokenize", "WhitespaceTokenizer", bench_whitespace_tokenizer),
    ("tokenize", "WordPunctTokenizer", bench_wordpunct_tokenizer),
    ("tokenize", "BlanklineTokenizer", bench_blankline_tokenizer),
    ("tokenize", "logos_word_tokenize", bench_logos_tokenizer),
    # stem (8)
    ("stem", "SnowballStemmer", bench_snowball),
    ("stem", "PorterStemmer", bench_porter),
    ("stem", "LancasterStemmer", bench_lancaster),
    ("stem", "WordNetLemmatizer", bench_wordnet),
    ("stem", "ARLSTem", bench_arlstem),
    ("stem", "ISRIStemmer", bench_isri_stemmer),
    ("stem", "RSLPStemmer", bench_rslp_stemmer),
    ("stem", "RegexpStemmer", bench_regexp_stemmer),
    # tag (9)
    ("tag", "PerceptronTagger", bench_perceptron_tagger),
    ("tag", "HMM tagger", bench_hmm_tag),
    ("tag", "TnT", bench_tnt_tag),
    ("tag", "DefaultTagger", bench_default_tagger),
    ("tag", "UnigramTagger", bench_unigram_tagger),
    ("tag", "BigramTagger", bench_bigram_tagger),
    ("tag", "TrigramTagger", bench_trigram_tagger),
    ("tag", "RegexpTagger", bench_regexp_tagger),
    ("tag", "AffixTagger", bench_affix_tagger),
    # classify (4)
    ("classify", "NaiveBayesClassifier.train", bench_naivebayes_train),
    ("classify", "NaiveBayesClassifier.classify", bench_naivebayes_classify),
    ("classify", "MaxentClassifier.train", bench_maxent_train),
    ("classify", "TextCat.guess_language", bench_textcat),
    # probability (4)
    ("probability", "FreqDist", bench_freqdist),
    ("probability", "ConditionalFreqDist", bench_conditional_freqdist),
    ("probability", "LaplaceProbDist", bench_laplace_probdist),
    ("probability", "MLEProbDist", bench_mle_probdist),
    # collocations (3)
    ("collocations", "BigramCollocationFinder", bench_bigram_collocations),
    ("collocations", "TrigramCollocationFinder", bench_trigram_collocations),
    ("collocations", "QuadgramCollocationFinder", bench_quadgram_collocations),
    # sentiment (1)
    ("sentiment", "SentimentIntensityAnalyzer", bench_sentiment),
    # metrics (4)
    ("metrics", "windowdiff", bench_windowdiff),
    ("metrics", "pk", bench_pk),
    ("metrics", "edit_distance", bench_edit_distance),
    ("metrics", "BigramAssocMeasures", bench_bigram_assoc_measures),
    # lm (6)
    ("lm", "MLE.score", bench_mle_fit),
    ("lm", "Lidstone", bench_lidstone),
    ("lm", "Laplace", bench_laplace_lm),
    ("lm", "StupidBackoff", bench_stupid_backoff),
    ("lm", "KneserNey", bench_kneser_ney),
    ("lm", "WittenBell", bench_witten_bell),
    # ccg (1)
    ("ccg", "CCG from_string", bench_ccg_parse),
    # chunk (1)
    ("chunk", "RegexpParser", bench_chunk_parse),
    # cluster (1)
    ("cluster", "KMeansClusterer", bench_kmeans),
    # parse (2)
    ("parse", "EarleyChartParser", bench_earley),
    ("parse", "CFG", bench_cfg),
    # translate (1)
    ("translate", "bleu", bench_bleu),
    # chat (1)
    ("chat", "Chat", bench_chat_respond),
    # tree (1)
    ("tree", "Tree.from_string", bench_tree_from_string),
    # sem (1)
    ("sem", "Expression", bench_sem_expr),
    # inference (4)
    ("inference", "Tableau prover", bench_tableau),
    ("inference", "Resolution prover", bench_resolution),
    ("inference", "Discourse QA", bench_discourse),
    ("inference", "DefaultReasoner", bench_nonmonotonic),
]


def run_all() -> list[BenchResult]:
    results: list[BenchResult] = []
    print(f"\n  Running {len(ALL_BENCHMARKS)} benchmarks...")
    for group, name, fn in ALL_BENCHMARKS:
        print(f"  \u2022 {group}/{name}...", end=" ", flush=True)
        try:
            r = fn()
            results.append(r)
            if r.speedup:
                print(f"{r.speedup:.1f}x")
            else:
                print(f"{r.fast_only_ms:.4f}ms")
        except Exception as e:
            print(f"FAILED \u2014 {e}")
    return results
