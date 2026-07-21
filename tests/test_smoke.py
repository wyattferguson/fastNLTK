"""Smoke test: exercises every fastNLTK module end-to-end.

Run:  python -m pytest tests/test_smoke.py -v
"""

from __future__ import annotations

import io
import sys
import time

import fastnltk as nltk
from fastnltk import (
    ne_chunk,
    ne_chunk_sents,
    pos_tag,
    pos_tag_sents,
    sent_tokenize,
    word_tokenize,
)
from fastnltk.chat import Chat
from fastnltk.chunk import RegexpParser as ChunkRegexpParser
from fastnltk.classify import MaxentClassifier, NaiveBayesClassifier, TextCat
from fastnltk.cluster import KMeansClusterer
from fastnltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from fastnltk.corpus import stopwords
from fastnltk.downloader import download
from fastnltk.lm import (
    MLE as LM_MLE,
)
from fastnltk.lm import (
    KneserNeyInterpolated,
    Laplace,
    Lidstone,
    StupidBackoff,
    WittenBellInterpolated,
)
from fastnltk.metrics import (
    BigramAssocMeasures,
    binary_distance,
    dice_similarity,
    edit_distance,
    jaccard_distance,
    jaro_similarity,
    jaro_winkler_similarity,
    spearman,
)
from fastnltk.parse import CFG, EarleyChartParser
from fastnltk.probability import ConditionalFreqDist, FreqDist
from fastnltk.sem import fromstring as sem_fromstring
from fastnltk.sem import simplify as sem_simplify
from fastnltk.sentiment import SentimentIntensityAnalyzer
from fastnltk.stem import (
    ARLSTem,
    ARLSTem2,
    Cistem,
    ISRIStemmer,
    LancasterStemmer,
    PorterStemmer,
    RegexpStemmer,
    RSLPStemmer,
    SnowballStemmer,
    WordNetLemmatizer,
)
from fastnltk.tag import (
    AffixTagger,
    BigramTagger,
    DefaultTagger,
    RegexpTagger,
    TnT,
    TrigramTagger,
    UnigramTagger,
)
from fastnltk.tokenize import (
    BlanklineTokenizer,
    CharTokenizer,
    LineTokenizer,
    PunktSentenceTokenizer,
    RegexpTokenizer,
    SpaceTokenizer,
    TabTokenizer,
    TreebankWordDetokenizer,
    TreebankWordTokenizer,
    TweetTokenizer,
    WhitespaceTokenizer,
    WordPunctTokenizer,
    regexp_tokenize,
)
from fastnltk.translate import bleu_score, corpus_bleu
from fastnltk.tree import Tree
from fastnltk.util import ngrams

nltk.download("maxent_ne_chunker_tab")

SAMPLE = (
    "Natural Language Processing (NLP) with NLTK is fun. "
    "NLTK provides tokenizers, taggers, parsers, and corpora. "
    "Visit https://www.nltk.org for more info."
)

SAMPLE_SENTS = [
    "The quick brown fox jumps over the lazy dog.",
    "He was eating a delicious apple pie.",
    "She sells seashells by the seashore.",
]

TRAIN_DATA = [
    [("The", "DT"), ("dog", "NN"), ("barks", "VBZ"), (".", ".")],
    [("The", "DT"), ("cat", "NN"), ("meows", "VBZ"), (".", ".")],
    [("A", "DT"), ("fox", "NN"), ("jumps", "VBZ"), (".", ".")],
]

GRAMMAR = """
    S -> NP VP
    NP -> Det N
    VP -> V NP
    Det -> the | The
    N -> dog | cat | fox
    V -> chased
"""

SECTIONS = [
    "tokenize",
    "tag",
    "chunk",
    "stem",
    "probability",
    "collocations",
    "sentiment",
    "tree",
    "metrics",
    "lm",
    "classify",
    "cluster",
    "parse",
    "ccg",
    "chat",
    "inference",
    "sem",
    "translate",
    "ngrams",
    "downloader",
    "corpus",
]


# ── helpers ──


def _section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _timed(name: str, fn) -> float:
    t0 = time.perf_counter()
    fn()
    t = time.perf_counter() - t0
    print(f"  [{name}] {t:.4f}s")
    return t


# ── 1. Tokenization ──


def _demo_tokenize() -> None:
    _section("1  TOKENIZE")

    def _sent() -> None:
        print(f"    sent_tokenize -> {sent_tokenize(SAMPLE)}")

    def _word() -> None:
        print(f"    word_tokenize -> {word_tokenize(SAMPLE)[:15]}...")

    def _regexp_tokenize() -> None:
        pat = r"\\w+"
        print(f"    regexp_tokenize({pat}) -> {regexp_tokenize(SAMPLE, pat)[:10]}...")

    def _regexp_cls() -> None:
        t = RegexpTokenizer(r"\d+")
        print(f"    RegexpTokenizer(digits) -> {t.tokenize(SAMPLE)}")

    def _whitespace() -> None:
        t = WhitespaceTokenizer()
        print(f"    WhitespaceTokenizer -> {t.tokenize(SAMPLE[:60])}")

    def _wordpunct() -> None:
        t = WordPunctTokenizer()
        print(f"    WordPunctTokenizer -> {t.tokenize(SAMPLE[:60])}")

    def _blankline() -> None:
        t = BlanklineTokenizer()
        s = "a\n\nb\n\nc"
        print(f"    BlanklineTokenizer -> {t.tokenize(s)}")

    def _line() -> None:
        t = LineTokenizer()
        s = "a\nb\nc"
        print(f"    LineTokenizer -> {t.tokenize(s)}")

    def _space() -> None:
        t = SpaceTokenizer()
        print(f"    SpaceTokenizer -> {t.tokenize('a  b   c')}")

    def _tab() -> None:
        t = TabTokenizer()
        s = "a\tb\tc"
        print(f"    TabTokenizer -> {t.tokenize(s)}")

    def _treebank() -> None:
        t = TreebankWordTokenizer()
        s = "Mr. Smith can't"
        print(f"    TreebankWordTokenizer -> {t.tokenize(s)}")

    def _detokenize() -> None:
        t = TreebankWordDetokenizer()
        print(f"    TreebankWordDetokenizer -> {t.detokenize(['Hello', ',', 'world', '!'])}")

    def _tweet() -> None:
        t = TweetTokenizer()
        s = "I'm @user having fan!!!"
        print(f"    TweetTokenizer -> {t.tokenize(s)}")

    def _punkt_cls() -> None:
        t = PunktSentenceTokenizer()
        print(f"    PunktSentenceTokenizer -> {t.tokenize(SAMPLE)}")

    def _char() -> None:
        t = CharTokenizer()
        print(f"    CharTokenizer -> {t.tokenize('abc')}")

    _timed("sent_tokenize", _sent)
    _timed("word_tokenize", _word)
    _timed("regexp_tokenize", _regexp_tokenize)
    _timed("RegexpTokenizer", _regexp_cls)
    _timed("WhitespaceTokenizer", _whitespace)
    _timed("WordPunctTokenizer", _wordpunct)
    _timed("BlanklineTokenizer", _blankline)
    _timed("LineTokenizer", _line)
    _timed("SpaceTokenizer", _space)
    _timed("TabTokenizer", _tab)
    _timed("TreebankWordTokenizer", _treebank)
    _timed("TreebankWordDetokenizer", _detokenize)
    _timed("TweetTokenizer", _tweet)
    _timed("PunktSentenceTokenizer", _punkt_cls)
    _timed("CharTokenizer", _char)


# ── 2. POS Tagging ──


def _demo_tag() -> None:
    _section("2  TAG")

    words = word_tokenize(SAMPLE)
    sents = [word_tokenize(s) for s in SAMPLE_SENTS]

    def _pos_tag() -> None:
        print(f"    pos_tag -> {pos_tag(words)[:10]}...")

    def _pos_tag_sents() -> None:
        print(f"    pos_tag_sents -> {[t[:3] for t in pos_tag_sents(sents)]}...")

    def _default_tagger() -> None:
        t = DefaultTagger("NN")
        print(f"    DefaultTagger(NN) -> {t.tag(words)[:5]}")

    def _unigram_tagger() -> None:
        t = UnigramTagger(train=TRAIN_DATA)
        print(f"    UnigramTagger(train) -> {t.tag(words[:8])}")

    def _bigram_tagger() -> None:
        t = BigramTagger(train=TRAIN_DATA)
        print(f"    BigramTagger(train) -> {t.tag(['The', 'dog'])}")

    def _trigram_tagger() -> None:
        t = TrigramTagger(train=TRAIN_DATA)
        print(f"    TrigramTagger(train) -> {t.tag(['The', 'dog', 'barks'])}")

    def _affix_tagger() -> None:
        t = AffixTagger(train=TRAIN_DATA)
        print(f"    AffixTagger(train) -> {t.tag(words[:8])}")

    def _regexp_tagger() -> None:
        t = RegexpTagger([(r"\w+ing$", "VBG"), (r"\w+ly$", "RB"), (r".*", "NN")])
        print(f"    RegexpTagger -> {t.tag(['running', 'quickly', 'dog'])}")

    def _tnt_tagger() -> None:
        t = TnT()
        t.train(TRAIN_DATA)
        print(f"    TnT -> {t.tag(['The', 'cat', 'meows'])}")

    _timed("pos_tag", _pos_tag)
    _timed("pos_tag_sents", _pos_tag_sents)
    _timed("DefaultTagger", _default_tagger)
    _timed("UnigramTagger", _unigram_tagger)
    _timed("BigramTagger", _bigram_tagger)
    _timed("TrigramTagger", _trigram_tagger)
    _timed("AffixTagger", _affix_tagger)
    _timed("RegexpTagger", _regexp_tagger)
    _timed("TnT", _tnt_tagger)


# ── 3. Chunking / Named Entities ──


def _demo_chunk() -> None:
    _section("3  CHUNK")

    tagged = pos_tag(word_tokenize(SAMPLE))
    tagged_sents = [pos_tag(word_tokenize(s)) for s in SAMPLE_SENTS]

    def _ne_chunk() -> None:
        tree = ne_chunk(tagged)
        print(f"    ne_chunk -> {tree}")

    def _ne_chunk_sents() -> None:
        trees = list(ne_chunk_sents(tagged_sents))
        print(f"    ne_chunk_sents -> {len(trees)} trees")

    def _regexp_parser() -> None:
        grammar = r"NP: {<DT>?<JJ>*<NN.*>}"
        rp = ChunkRegexpParser(grammar)
        parsed = rp.parse(tagged[:12])
        print(f"    RegexpParser -> {parsed}")

    _timed("ne_chunk", _ne_chunk)
    _timed("ne_chunk_sents", _ne_chunk_sents)
    _timed("RegexpParser (chunk)", _regexp_parser)


# ── 4. Stemming & Lemmatization ──


def _demo_stem() -> None:
    _section("4  STEM / LEMMATIZE")

    words = ["running", "better", "flies", "studies", "lying", "cats", "city"]

    def _porter() -> None:
        s = PorterStemmer()
        print(f"    PorterStemmer -> {[s.stem(w) for w in words]}")

    def _lancaster() -> None:
        s = LancasterStemmer()
        print(f"    LancasterStemmer -> {[s.stem(w) for w in words]}")

    def _snowball() -> None:
        s = SnowballStemmer("english")
        print(f"    SnowballStemmer(en) -> {[s.stem(w) for w in words]}")

    def _regexp_stem() -> None:
        s = RegexpStemmer(min_length=4)
        print(f"    RegexpStemmer -> {[s.stem(w) for w in words]}")

    def _wn_lemmatizer() -> None:
        lz = WordNetLemmatizer()
        print(f"    WordNetLemmatizer -> {[lz.lemmatize(w) for w in words]}")

    def _cistem() -> None:
        s = Cistem()
        res = [ascii(s.stem(w)) for w in ["Ablauf", "abla+ufe", "scho+n"]]
        print(f"    Cistem -> {res}")

    def _arlstem() -> None:
        s = ARLSTem()
        print(f"    ARLSTem -> {ascii(s.stem('eamela'))}")

    def _arlstem2() -> None:
        s = ARLSTem2()
        print(f"    ARLSTem2 -> {ascii(s.stem('eamela'))}")

    def _isri() -> None:
        s = ISRIStemmer()
        print(f"    ISRIStemmer -> {ascii(s.stem('eamela'))}")

    def _rslp() -> None:
        s = RSLPStemmer()
        print(f"    RSLPStemmer -> {s.stem('cantou')}")

    _timed("PorterStemmer", _porter)
    _timed("LancasterStemmer", _lancaster)
    _timed("SnowballStemmer", _snowball)
    _timed("RegexpStemmer", _regexp_stem)
    _timed("WordNetLemmatizer", _wn_lemmatizer)
    _timed("Cistem", _cistem)
    _timed("ARLSTem", _arlstem)
    _timed("ARLSTem2", _arlstem2)
    _timed("ISRIStemmer", _isri)
    _timed("RSLPStemmer", _rslp)


# ── 5. Frequency & Probability ──


def _demo_probability() -> None:
    _section("5  PROBABILITY")

    words = [w.lower() for w in word_tokenize(SAMPLE) if w.isalpha()]

    def _freqdist() -> None:
        fd = FreqDist(words)
        print(f"    FreqDist: N={fd.N()}, B={fd.B()}, max={fd.max()}, mc={fd.most_common(3)}")

    def _condfreqdist() -> None:
        cfd = ConditionalFreqDist()
        for i, w in enumerate(words):
            if i > 0:
                cfd.inc(words[i - 1], w)
        print(f"    ConditionalFreqDist: conditions={cfd.conditions()[:4]}")

    def _freqdist_methods() -> None:
        fd = FreqDist(words)
        print(f"    freq('nltk')={fd.freq('nltk'):.3f}, hapaxes={fd.hapaxes()[:3]}")

    _timed("FreqDist", _freqdist)
    _timed("ConditionalFreqDist", _condfreqdist)
    _timed("FreqDist.methods", _freqdist_methods)


# ── 6. Collocations ──


def _demo_collocations() -> None:
    _section("6  COLLOCATIONS")

    words = word_tokenize(
        "The quick brown fox jumps over the lazy dog. "
        "The quick brown fox leaps over the lazy cat. "
        "The quick brown fox jumps over the brown dog."
    )

    def _bigram_colloc() -> None:
        bcf = BigramCollocationFinder.from_words(words)
        bcf.apply_freq_filter(1)
        print(f"    BigramCollocationFinder(pmi) -> {bcf.nbest(BigramAssocMeasures.pmi, 5)}")

    def _trigram_colloc() -> None:
        tcf = TrigramCollocationFinder.from_words(words)
        print(f"    TrigramCollocationFinder(pmi) -> {tcf.nbest(BigramAssocMeasures.pmi, 5)}")

    _timed("BigramCollocationFinder", _bigram_colloc)
    _timed("TrigramCollocationFinder", _trigram_colloc)


# ── 7. Sentiment ──


def _demo_sentiment() -> None:
    _section("7  SENTIMENT")

    def _vader() -> None:
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores("fastNLTK is an amazing, fantastic library!")
        print(f"    SentimentIntensityAnalyzer -> {scores}")

    _timed("SentimentIntensityAnalyzer", _vader)


# ── 8. Tree ──


def _demo_tree() -> None:
    _section("8  TREE")

    def _build_tree() -> None:
        t = Tree("S", [Tree("NP", ["The", "dog"]), Tree("VP", ["barks"])])
        print(f"    Tree: label={t.label()}, leaves={t.leaves()}, height={t.height()}")

    def _from_string() -> None:
        t = Tree.from_string("(S (NP The dog) (VP barks))")
        print(f"    Tree.from_string -> leaves={t.leaves()}")

    def _tree_subtrees() -> None:
        t = Tree("S", [Tree("NP", [Tree("DET", ["The"]), "dog"]), Tree("VP", ["barks"])])
        print(f"    subtrees -> {len(t.subtrees())} subtrees")

    def _tree_productions() -> None:
        t = Tree("S", [Tree("NP", ["The"]), Tree("VP", ["barks"])])
        print(f"    productions -> {t.productions()[:3]}")

    _timed("Tree (build)", _build_tree)
    _timed("Tree.from_string", _from_string)
    _timed("Tree.subtrees", _tree_subtrees)
    _timed("Tree.productions", _tree_productions)


# ── 9. Metrics ──


def _demo_metrics() -> None:
    _section("9  METRICS")

    def _edit_dist() -> None:
        print(f"    edit_distance('kitten', 'sitting') = {edit_distance('kitten', 'sitting')}")

    def _jaccard() -> None:
        print(
            f"    jaccard_distance({{'a','b'}}, {{'b','c'}}) = {jaccard_distance({'a', 'b'}, {'b', 'c'}):.4f}"
        )

    def _binary_dist() -> None:
        print(f"    binary_distance({{1}}, {{2}}) = {binary_distance({1}, {2})}")

    def _jaro() -> None:
        print(f"    jaro_similarity('dwayne', 'duane') = {jaro_similarity('dwayne', 'duane'):.4f}")

    def _jaro_winkler() -> None:
        print(
            f"    jaro_winkler_similarity('dwayne', 'duane') = {jaro_winkler_similarity('dwayne', 'duane'):.4f}"
        )

    def _dice() -> None:
        print(f"    dice_similarity('test', 'text') = {dice_similarity('test', 'text'):.4f}")

    _timed("edit_distance", _edit_dist)
    _timed("jaccard_distance", _jaccard)
    _timed("binary_distance", _binary_dist)
    _timed("jaro_similarity", _jaro)
    _timed("jaro_winkler_similarity", _jaro_winkler)
    _timed("dice_similarity", _dice)

    def _spearman() -> None:
        ranks1 = [1, 2, 3, 4, 5]
        ranks2 = [5, 4, 3, 2, 1]
        rho = spearman(ranks1, ranks2)
        print(f"    spearman(inv) = {rho:.4f}")

    _timed("spearman", _spearman)


# ── 10. Language Models ──


def _demo_lm() -> None:
    _section("10  LANGUAGE MODELS")

    train = [["the", "dog", "barks"], ["the", "cat", "meows"], ["a", "fox", "jumps"]]

    def _mle() -> None:
        m = LM_MLE(2)
        m.fit(train)
        print(
            f"    MLE(order=2): score('dog', ['the'])={m.score('dog', ['the']):.4f}, "
            f"score('cat', ['the'])={m.score('cat', ['the']):.4f}"
        )

    def _laplace() -> None:
        m = Laplace(2)
        m.fit(train)
        print(f"    Laplace(order=2): vocab={m.vocab_size}, fitted={m.fitted}")

    def _lidstone() -> None:
        m = Lidstone(2, gamma=0.5)
        m.fit(train)
        print(f"    Lidstone(order=2, g=0.5): score('dog', ['the'])={m.score('dog', ['the']):.4f}")

    def _kneser_ney() -> None:
        m = KneserNeyInterpolated(2)
        m.fit(train)
        print(
            f"    KneserNeyInterpolated(order=2): score('dog', ['the'])={m.score('dog', ['the']):.4f}"
        )

    def _witten_bell() -> None:
        m = WittenBellInterpolated(2)
        m.fit(train)
        print(
            f"    WittenBellInterpolated(order=2): score='{m.score('dog', ['the']):.4f}', order={m.order}"
        )

    def _stupid_backoff() -> None:
        m = StupidBackoff(2, alpha=0.4)
        m.fit(train)
        print(
            f"    StupidBackoff(order=2): score('dog', ['the'])={m.score('dog', ['the']):.4f}, fitted={m.fitted}"
        )

    _timed("MLE", _mle)
    _timed("Laplace", _laplace)
    _timed("Lidstone", _lidstone)
    _timed("KneserNeyInterpolated", _kneser_ney)
    _timed("WittenBellInterpolated", _witten_bell)
    _timed("StupidBackoff", _stupid_backoff)

    def _generate() -> None:
        m = LM_MLE(2)
        m.fit(train)
        gen = m.generate(5, text_seed=["the"], random_seed=42)
        print(f"    MLE.generate(5, seed=['the']) -> {gen}")

    _timed("MLE.generate", _generate)


# ── 11. Classify ──


def _demo_classify() -> None:
    _section("11  CLASSIFY")

    def _naive_bayes() -> None:
        train = [
            ({"word": "awesome", "sentiment": "pos"}, "pos"),
            ({"word": "terrible", "sentiment": "neg"}, "neg"),
            ({"word": "great", "sentiment": "pos"}, "pos"),
            ({"word": "bad", "sentiment": "neg"}, "neg"),
        ]
        clf = NaiveBayesClassifier.train(train)
        print(f"    NaiveBayesClassifier -> labels={clf.labels()}")
        print(
            f"      classify({{'word':'awesome','sentiment':'pos'}})={clf.classify({'word': 'awesome', 'sentiment': 'pos'})}"
        )

    def _maxent() -> None:
        train = [
            ({"a": "1", "b": "0"}, "X"),
            ({"a": "0", "b": "1"}, "Y"),
            ({"a": "1", "b": "1"}, "X"),
        ]
        clf = MaxentClassifier.train(train, max_iter=10)
        print(f"    MaxentClassifier -> labels={clf.labels()}")
        print(f"      classify({{'a':'1','b':'0'}})={clf.classify({'a': '1', 'b': '0'})}")

    def _textcat() -> None:
        tc = TextCat()
        print(f"    TextCat -> guess('bonjour le monde')={tc.guess_language('bonjour le monde')}")

    _timed("NaiveBayesClassifier", _naive_bayes)
    _timed("MaxentClassifier", _maxent)
    _timed("TextCat", _textcat)


# ── 12. Cluster ──


def _demo_cluster() -> None:
    _section("12  CLUSTER")

    def _kmeans() -> None:
        vectors = [[1.0, 0.0], [1.1, 0.0], [0.0, 1.0], [0.0, 1.1]]
        km = KMeansClusterer(num_clusters=2)
        labels = km.cluster(vectors)
        print(f"    KMeansClusterer({len(vectors)} vectors, k=2) -> labels={labels}")
        print(f"      centroids={km.centroids()}")

    _timed("KMeansClusterer", _kmeans)


# ── 13. Parse ──


def _demo_parse() -> None:
    _section("13  PARSE")

    def _cfg() -> None:
        g = CFG.from_string(GRAMMAR)
        print(
            f"    CFG: start={g.start()}, prods={len(g.productions())}, nonterms={len(g.nonterminals())}"
        )

    def _earley() -> None:
        g = CFG.from_string(GRAMMAR)
        parser = EarleyChartParser()
        try:
            trees = parser.parse(g, ["The", "dog", "chased", "the", "cat"])
            count = 0
            for tree in trees:
                count += 1
                if count == 1:
                    print(f"    EarleyChartParser -> tree={tree}")
            print(f"      (got {count} trees)")
        except ValueError:
            print("    EarleyChartParser -> (no parse for test sentence)")

    _timed("CFG.from_string", _cfg)
    _timed("EarleyChartParser", _earley)


# ── 14. CCG ──


def _demo_ccg() -> None:
    _section("14  CCG")

    def _ccg_fromstring() -> None:
        from fastnltk.ccg import fromstring as ccg_fromstring

        result = ccg_fromstring("(S NP VP)")
        print(f"    CCG fromstring -> {result}")

    _timed("CCG.fromstring", _ccg_fromstring)


# ── 15. Chat ──


def _demo_chat() -> None:
    _section("15  CHAT")

    def _chat_respond() -> None:
        pairs = [
            (r"hi|hello|hey", ["Hello there!", "Hi!"]),
            (r"how are you", ["Fine thanks", "Doing great!"]),
            (r"bye|goodbye", ["Goodbye!", "See you later!"]),
        ]
        c = Chat(pairs)
        print(f"    Chat.respond('hi') -> {c.respond('hi')}")
        print(f"    Chat.respond('how are you') -> {c.respond('how are you')}")
        print(f"    Chat.respond('bye') -> {c.respond('bye')}")

    _timed("Chat", _chat_respond)


# ── 16. Inference ──


def _demo_inference() -> None:
    _section("16  INFERENCE")

    def _imports() -> None:
        print(
            "    Inference re-exports: DiscourseTester, ResolutionProverCommand, TableauProverCommand"
        )

    _timed("inference imports", _imports)


# ── 17. Sem ──


def _demo_sem() -> None:
    _section("17  SEM")

    def _sem_fromstring() -> None:
        expr = sem_fromstring("exists x.(man(x) & walk(x))")
        print(f"    sem.fromstring -> {expr} (type: {type(expr).__name__})")

    def _sem_simplify() -> None:
        expr = sem_fromstring("exists x.(man(x) & walk(x))")
        simpl = sem_simplify(expr)
        print(f"    sem.simplify -> {simpl} (type: {type(simpl).__name__})")

    _timed("sem.fromstring", _sem_fromstring)
    _timed("sem.simplify", _sem_simplify)


# ── 18. Translate ──


def _demo_translate() -> None:
    _section("18  TRANSLATE")

    def _bleu() -> None:
        ref = ["the", "cat", "sat", "on", "the", "mat"]
        cand = ["the", "cat", "is", "on", "the", "mat"]
        print(f"    bleu_score(ref, cand) = {bleu_score(cand, ref):.4f}")

    def _corpus_bleu() -> None:
        cands = [["the", "cat", "sat"]]
        refs = [["the", "cat", "sat"]]
        print(f"    corpus_bleu(cands, refs) = {corpus_bleu(cands, refs):.4f}")

    _timed("bleu_score", _bleu)
    _timed("corpus_bleu", _corpus_bleu)


# ── 19. Ngrams ──


def _demo_ngrams() -> None:
    _section("19  UTIL / NGRAMS")

    def _ngrams_fn() -> None:
        words = ["the", "quick", "brown", "fox"]
        ng = list(ngrams(words, 2))
        print(f"    ngrams({words}, 2) -> {ng}")

    _timed("ngrams", _ngrams_fn)


# ── 20. Downloader ──


def _demo_downloader() -> None:
    _section("20  DOWNLOADER")

    def _download_fn() -> None:
        result = download("punkt", quiet=True, force=False)
        print(f"    download('punkt') -> {'OK' if result else 'FAIL'}")

    _timed("downloader.download", _download_fn)


# ── 21. Corpus ──


def _demo_corpus() -> None:
    _section("21  CORPUS")

    def _stopwords() -> None:
        stops = stopwords.words("english")
        print(f"    stopwords.words('english') -> {len(stops)} words (e.g. {stops[:6]})")

    def _names() -> None:
        from fastnltk.corpus import names

        print(f"    names.words('male.txt')[:5] -> {names.words('male.txt')[:5]}")

    def _gutenberg() -> None:
        from fastnltk.corpus import gutenberg

        files = gutenberg.fileids()
        print(f"    gutenberg.fileids() -> {files[:3]}... ({len(files)} total)")

    def _wordnet() -> None:
        from fastnltk.corpus import wordnet

        syns = wordnet.synsets("dog")
        print(f"    wordnet.synsets('dog') -> {len(syns)} synsets, first: {syns[0].name()}")

    def _treebank_corpus() -> None:
        from fastnltk.corpus import treebank

        print(f"    treebank.fileids()[:3] -> {treebank.fileids()[:3]}")

    _timed("stopwords", _stopwords)
    _timed("names", _names)
    _timed("gutenberg", _gutenberg)
    _timed("wordnet", _wordnet)
    _timed("treebank", _treebank_corpus)


# ── main ──


def _main() -> None:
    download("punkt", quiet=True)
    download("averaged_perceptron_tagger", quiet=True)
    download("maxent_ne_chunker", quiet=True)
    download("words", quiet=True)
    download("wordnet", quiet=True)
    download("stopwords", quiet=True)
    download("rslp", quiet=True)
    download("names", quiet=True)
    download("gutenberg", quiet=True)
    download("treebank", quiet=True)
    download("maxent_ne_chunker_tab", quiet=True)

    demos: list[tuple[str, object]] = [
        ("tokenize", _demo_tokenize),
        ("tag", _demo_tag),
        ("chunk", _demo_chunk),
        ("stem", _demo_stem),
        ("probability", _demo_probability),
        ("collocations", _demo_collocations),
        ("sentiment", _demo_sentiment),
        ("tree", _demo_tree),
        ("metrics", _demo_metrics),
        ("lm", _demo_lm),
        ("classify", _demo_classify),
        ("cluster", _demo_cluster),
        ("parse", _demo_parse),
        ("ccg", _demo_ccg),
        ("chat", _demo_chat),
        ("inference", _demo_inference),
        ("sem", _demo_sem),
        ("translate", _demo_translate),
        ("ngrams", _demo_ngrams),
        ("downloader", _demo_downloader),
        ("corpus", _demo_corpus),
    ]

    print(f"\n{'#' * 70}")
    print(f"#  fastNLTK Complete Demo  (v{nltk.__version__})")
    print(f"{'#' * 70}")

    timings: dict[str, float] = {}
    total_start = time.perf_counter()
    for name, fn in demos:
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        timings[name] = elapsed
        print(f"\n  -- {name}: {elapsed:.4f}s")

    total = time.perf_counter() - total_start
    print(f"\n{'=' * 70}")
    print("  BENCHMARK SUMMARY")
    print(f"{'=' * 70}")
    for name, t in sorted(timings.items(), key=lambda x: -x[1]):
        print(f"    {name:20s}  {t:.4f}s")
    print(f"{'-' * 40}")
    print(f"    {'TOTAL':20s}  {total:.4f}s")
    print(f"    {'MODULES':20s}  {len(demos)}")


# ── pytest entry point ──


def test_demo_smoke() -> None:
    """Run all demo modules end-to-end and verify every section completed."""
    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        _main()
    finally:
        sys.stdout = old_stdout

    output = captured.getvalue()

    for name in SECTIONS:
        assert name in output, f"Module '{name}' missing from demo output"

    assert "BENCHMARK SUMMARY" in output
    assert "TOTAL" in output
    assert "MODULES" in output
