"""
Benchmark suite — all benchmark definitions for fastNLTK.

Each function returns a `BenchResult` or list of `BenchResult`.
"""

import json

from .harness import (
    BenchResult,
    _median_time,
    fixture,
)

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
        nltk_ms=n_ms, fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0, iterations=30,
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
        nltk_ms=n_ms, fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0, iterations=30,
    )


def bench_texttiling() -> BenchResult:
    from fastnltk._rust import TextTilingTokenizer

    text = fixture("medium")
    tt = TextTilingTokenizer(20, 10, True)
    f_ms = _median_time(lambda: tt.tokenize(text), 10)
    return BenchResult(
        name="TextTilingTokenizer.tokenize",
        group="tokenize",
        params={"chars": len(text)},
        fast_only_ms=f_ms, iterations=10,
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
        name="windowdiff", group="metrics",
        params={"chars": len(s1)},
        nltk_ms=n_ms, fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0, iterations=100,
    )


def bench_pk() -> BenchResult:
    from nltk.metrics.segmentation import pk as npk
    from fastnltk._rust import pk as fpk

    s1 = "000100000010" * 1000
    s2 = "000010000100" * 1000
    n_ms = _median_time(lambda: npk(s1, s2, 3, "1"), 100)
    f_ms = _median_time(lambda: fpk(s1, s2, 3, "1"), 100)
    return BenchResult(
        name="pk", group="metrics",
        params={"chars": len(s1)},
        nltk_ms=n_ms, fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0, iterations=100,
    )


# ── HMM Tagger ──────────────────────────────────────────────


def bench_hmm_tag() -> BenchResult:
    from fastnltk._rust import HiddenMarkovModelTagger

    train = [[("I", "PRP"), ("like", "VBP"), ("dogs", "NNS")],
             [("She", "PRP"), ("runs", "VBZ"), ("fast", "RB")]]
    hm = HiddenMarkovModelTagger(3, 5, 1e-4, 0.1)
    hm.train(train)
    words_1k = ["I", "like", "dogs"] * 333
    f_ms = _median_time(lambda: hm.tag(words_1k), 50)
    return BenchResult(
        name="HiddenMarkovModelTagger.tag",
        group="tag",
        params={"words": len(words_1k)},
        fast_only_ms=f_ms, iterations=50,
    )


# ── LM Scores ─────────────────────────────────────────────


def bench_kneser_ney() -> BenchResult:
    from fastnltk.lm import KneserNeyInterpolated

    def run():
        m = KneserNeyInterpolated(2, 0.75)
        m.fit([["the", "cat"], ["the", "dog"], ["a", "cat"], ["the", "mouse"]])
        return [m.score(w, ["the"]) for w in ["cat", "dog", "mouse", "rat"]]

    f_ms = _median_time(run, 100)
    return BenchResult(
        name="KneserNeyInterpolated.score",
        group="lm",
        params={"queries": 4},
        fast_only_ms=f_ms, iterations=100,
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
        name="CCG from_string", group="ccg",
        params={"parses": len(cats)},
        nltk_ms=n_ms, fast_ms=f_ms,
        speedup=n_ms / f_ms if f_ms else 0, iterations=30,
    )


# ── Inference ─────────────────────────────────────────────


def bench_tableau() -> BenchResult:
    from fastnltk._rust import TableauProver

    tp = TableauProver(200)
    f_ms = _median_time(lambda: tp.prove("P | ~P", None), 50)
    return BenchResult(
        name="TableauProver.prove", group="inference",
        params={"formula": "P | ~P"},
        fast_only_ms=f_ms, iterations=50,
    )


def bench_resolution() -> BenchResult:
    from fastnltk._rust import ResolutionProver

    rp = ResolutionProver(1000)
    f_ms = _median_time(lambda: rp.prove("P | ~P", None), 50)
    return BenchResult(
        name="ResolutionProver.prove", group="inference",
        params={"formula": "P | ~P"},
        fast_only_ms=f_ms, iterations=50,
    )


def bench_discourse() -> BenchResult:
    from fastnltk._rust import DiscourseThread

    dt = DiscourseThread()
    dt.add_drs("([x],[dog(x)])")
    dt.add_drs("([y],[cat(y)])")
    val = json.dumps({"dog": [["fido"]], "cat": [["felix"]]})
    dom = json.dumps(["fido", "felix"])
    f_ms = _median_time(
        lambda: dt.answer_question("([x],[dog(x)])", val, dom), 50
    )
    return BenchResult(
        name="DiscourseThread.answer_question", group="inference",
        params={},
        fast_only_ms=f_ms, iterations=50,
    )


def bench_nonmonotonic() -> BenchResult:
    from fastnltk._rust import DefaultRule, DefaultReasoner

    rules = [DefaultRule("", f"fact{i}", f"fact{i}", "") for i in range(10)]
    r = DefaultReasoner(rules, 10)
    f_ms = _median_time(lambda: r.extensions(), 50)
    return BenchResult(
        name="DefaultReasoner.extensions", group="inference",
        params={"rules": 10},
        fast_only_ms=f_ms, iterations=50,
    )


# ── Registry ──────────────────────────────────────────────

ALL_BENCHMARKS: list[tuple[str, str, callable]] = [
    # (group, name, fn)
    ("tokenize", "ToktokTokenizer", bench_toktok),
    ("tokenize", "MWETokenizer", bench_mwe),
    ("tokenize", "TextTilingTokenizer", bench_texttiling),
    ("metrics", "windowdiff", bench_windowdiff),
    ("metrics", "pk", bench_pk),
    ("tag", "HMM tagger", bench_hmm_tag),
    ("lm", "KneserNey", bench_kneser_ney),
    ("ccg", "CCG from_string", bench_ccg_parse),
    ("inference", "Tableau prover", bench_tableau),
    ("inference", "Resolution prover", bench_resolution),
    ("inference", "Discourse QA", bench_discourse),
    ("inference", "DefaultReasoner", bench_nonmonotonic),
]


def run_all() -> list[BenchResult]:
    results: list[BenchResult] = []
    print(f"\n  Running {len(ALL_BENCHMARKS)} benchmarks...")
    for group, name, fn in ALL_BENCHMARKS:
        print(f"  • {group}/{name}...", end=" ", flush=True)
        try:
            r = fn()
            results.append(r)
            sp = r.speedup if r.speedup else r.fast_only_ms
            if r.speedup:
                print(f"{r.speedup:.1f}x")
            else:
                print(f"{r.fast_only_ms:.4f}ms")
        except Exception as e:
            print(f"FAILED — {e}")
    return results
