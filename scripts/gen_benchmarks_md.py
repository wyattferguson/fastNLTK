#!/usr/bin/env python
"""Regenerate BENCHMARKS.md from a benchmarks/results JSON file."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JSON_FILE = sys.argv[1] if len(sys.argv) > 1 else None

if JSON_FILE is None:
    results_dir = ROOT / "benchmarks" / "results"
    jsons = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not jsons:
        sys.exit("no results json found")
    JSON_FILE = str(jsons[-1])

data = json.loads(Path(JSON_FILE).read_text(encoding="utf-8"))
rs = [r for r in data["results"] if r.get("nltk_ms", 0) > 0]
all_r = data["results"]


def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


gm = geo([r["speedup"] for r in rs])

# map result name -> display row name (strip trailing .method for some)
def display(name):
    return name


GROUPS = [
    ("tokenize", ["Toktok", "MWE", "RegexpTokenizer", "SpaceTokenizer", "TreebankWordTokenizer",
                  "TweetTokenizer", "TextTiling", "SExpr", "Punkt", "Detokenizer", "TabTokenizer",
                  "LineTokenizer", "WhitespaceTokenizer", "WordPunctTokenizer", "BlanklineTokenizer", "logos"]),
    ("stem", ["Snowball", "Porter", "Lancaster", "WordNet", "ARLSTem", "ISRI", "RSLP", "RegexpStemmer"]),
    ("tag", ["Perceptron", "HiddenMarkov", "TnT", "DefaultTagger", "UnigramTagger", "BigramTagger",
             "TrigramTagger", "RegexpTagger", "AffixTagger"]),
    ("classify", ["NaiveBayes", "Maxent", "TextCat"]),
    ("probability", ["FreqDist", "ConditionalFreqDist", "LaplaceProbDist", "MLEProbDist"]),
    ("collocations", ["BigramCollocation", "TrigramCollocation", "QuadgramCollocation"]),
    ("sentiment", ["SentimentIntensityAnalyzer"]),
    ("metrics", ["windowdiff", "pk", "edit_distance", "BigramAssocMeasures"]),
    ("lm", ["MLE.score", "Lidstone.score", "Laplace.score", "StupidBackoff.score",
            "KneserNeyInterpolated.score", "WittenBellInterpolated.score"]),
    ("ccg", ["CCG"]),
    ("chunk", ["RegexpParser"]),
    ("cluster", ["KMeans"]),
    ("parse", ["Earley", "CFG"]),
    ("translate", ["bleu"]),
    ("chat", ["Chat"]),
    ("tree", ["Tree.from_string"]),
    ("sem", ["Expression.fromstring"]),
    ("inference", ["TableauProver", "ResolutionProver", "DiscourseThread", "DefaultReasoner"]),
]


def bold_speedup(s, fast_ms, nltk_ms):
    s = f"{s:.1f}x"
    return f"**{s}**" if (nltk_ms and fast_ms and s != "0.0x") else s


rows = {}
for r in all_r:
    rows[r["name"]] = r

lines = []
lines.append("# Benchmarks\n")
lines.append(f"> **Last updated:** 2026-08-05 (v0.5.5, release build)")
lines.append(f"> **Geometric mean: {gm:.1f}× vs NLTK** across {len(rs)} compared benchmarks ({len(all_r)} total).\n")
lines.append("> Run benchmarks: `python -m benchmarks.run --save`")
lines.append("> Fixtures: NLTK Gutenberg corpus (~200KB medium, ~5KB tiny).\n")
lines.append("> [!NOTE]")
lines.append("> HMM tagger optimized in v0.5.3: integer tag IDs + flat matrices —")
lines.append("> eliminates `String::clone()` in the O(N × T²) Viterbi inner loop (85× speedup).")
lines.append("> ConditionalFreqDist now shares `FreqDist` references so mutations via")
lines.append("> `cfd[cond][sample] = value` propagate correctly.")
lines.append("> ISRI and RSLP stemmers delegate to NLTK in the Python wrapper (`fastnltk.stem`)")
lines.append("> for byte-identical output. The raw Rust `_rust` versions are benchmarked below")
lines.append("> but the user-facing interface matches NLTK exactly.\n")
lines.append("---\n")
lines.append("## Highlights\n")
lines.append("| Operation | NLTK (ms) | fastNLTK (ms) | Speedup | Notes |")
lines.append("|---|---|---|---|---|")

NOTES = {
    "TextTilingTokenizer.tokenize": "SIMD memchr3 + byte-level segmentation",
    "MaxentClassifier.train": "GIS training, fully optimized inner loop",
    "windowdiff": "Pure algorithmic port, zero Python overhead",
    "edit_distance": "Damerau-Levenshtein in Rust",
    "HiddenMarkovModelTagger.tag": "Integer Viterbi, zero-alloc inner loop",
    "pk": "Segmentation metric in Rust",
    "TreebankWordDetokenizer.detokenize": "Single-pass undo",
    "SentimentIntensityAnalyzer.polarity_scores": "PHF lexicon, exact NLTK scoring",
    "PunktSentenceTokenizer.tokenize": "Byte-level sentence scan",
    "Expression.fromstring": "FOL parser in Rust",
    "SExprTokenizer.tokenize": "S-expression parser",
    "TweetTokenizer.tokenize": "LazyLock regexes",
    "CFG.from_string": "Grammar parser in Rust",
    "LancasterStemmer.stem": "Full 124-rule NLTK port",
    "QuadgramCollocationFinder.from_words": "FastMap ngram counting",
}
HIGHLIGHT_ORDER = [
    "TextTilingTokenizer.tokenize", "MaxentClassifier.train", "windowdiff",
    "edit_distance", "HiddenMarkovModelTagger.tag", "pk",
    "TreebankWordDetokenizer.detokenize", "SentimentIntensityAnalyzer.polarity_scores",
    "PunktSentenceTokenizer.tokenize", "Expression.fromstring", "SExprTokenizer.tokenize",
    "TweetTokenizer.tokenize", "CFG.from_string", "LancasterStemmer.stem",
    "QuadgramCollocationFinder.from_words",
]
for name in HIGHLIGHT_ORDER:
    r = rows[name]
    short = name.replace("Classifier", "").replace("Tokenizer", "").replace(".tokenize", "").replace(".tag", "")
    lines.append(
        f"| **{short}** | {r['nltk_ms']:.2f} | **{r['fast_ms']:.2f}** | **{r['speedup']:.0f}×** | {NOTES[name]} |"
    )

lines.append("")
lines.append("---\n")
lines.append(f"## Full Results ({len(all_r)} benchmarks)\n")
lines.append("Benchmarks grouped by module. Numbers from `python -m benchmarks.run --save` on release build.\n")
lines.append("| Module | Benchmark | NLTK (ms) | fastNLTK (ms) | Speedup |")
lines.append("|--------|-----------|-----------|---------------|---------|")

for gname, keys in GROUPS:
    lines.append(f"| **{gname}** | | | | |")
    # include every result whose name starts with any key in this group
    matched = [n for n in rows if any(n.startswith(k) for k in keys)]
    for name in matched:
        r = rows[name]
        nltk_ms = f"{r['nltk_ms']:.2f}" if r["nltk_ms"] else "—"
        fast_val = r["fast_ms"] if r["nltk_ms"] else r.get("fast_only_ms", 0.0)
        fast_ms = f"{fast_val:.4f}" if (not r["nltk_ms"] and fast_val < 0.01) else f"{fast_val:.2f}"
        if r["nltk_ms"]:
            speed = f"{r['speedup']:.1f}×"
            speed = f"**{speed}**" if r["speedup"] >= 1.5 else speed
        else:
            speed = "—"
        disp = name + (" †" if not r["nltk_ms"] else "")
        lines.append(f"| | {disp} | {nltk_ms} | {fast_ms} | {speed} |")

lines.append("")
lines.append("† fastNLTK-only — no NLTK comparison available.")
lines.append("")
lines.append("---\n")
lines.append("## Module Leaderboard\n")
lines.append("| Module | Geo Mean Speedup | Best Single | Key Engine |")
lines.append("|--------|-----------------|-------------|------------|")

MODULE_ENGINE = {
    "metrics": "Pure algorithmic port, zero Python overhead",
    "sentiment": "PHF lexicon, exact NLTK algorithm",
    "sem": "FOL expression parser",
    "classify": "GIS training, fully optimized inner loop",
    "collocations": "FastMap ngram frequency counting",
    "tree": "Tree bracket parser",
    "translate": "BLEU in Rust",
    "stem": "124-rule NLTK port",
    "chunk": "Regexp chunk parser",
    "tokenize": "SIMD memchr3 + char scanner + byte-level segmentation",
    "cluster": "K-means in Rust",
    "tag": "u64 feature IDs, integer Viterbi",
    "parse": "Earley + CFG parsing",
    "probability": "SmolStr-optimized FreqDist",
    "ccg": "CCG category parsing",
    "chat": "Eliza chatbot",
}

# compute per-module geo mean from actual group assignments
module_groups = {}
for r in rs:
    # derive module from name heuristically (same order as GROUPS)
    for gname, keys in GROUPS:
        if any(r["name"].startswith(k) for k in keys):
            module_groups.setdefault(gname, []).append(r["speedup"])
            break

leaderboard = []
for gname, speeds in module_groups.items():
    gm2 = geo(speeds)
    best = max(speeds)
    best_name = next(r["name"] for r in rs if r["speedup"] == best and any(r["name"].startswith(k) for k in dict(GROUPS)[gname]))
    leaderboard.append((gname, gm2, best, best_name))
leaderboard.sort(key=lambda x: -x[1])

for gname, gm2, best, best_name in leaderboard:
    short = best_name.replace("Classifier", "").replace("Tokenizer", "").replace(".tokenize", "").replace(".tag", "").replace(".polarity_scores", "").replace(".fromstring", "").replace(".from_words", "").replace(".stem", "")
    lines.append(f"| {gname} | **{gm2:.0f}×** | {best:.0f}× ({short}) | {MODULE_ENGINE[gname]} |")

out = "\n".join(lines) + "\n"
(ROOT / "BENCHMARKS.md").write_text(out, encoding="utf-8")
print(f"wrote BENCHMARKS.md (geo mean {gm:.1f}x)")
