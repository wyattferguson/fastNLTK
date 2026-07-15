"""Benchmark all remaining unbenched public API functions."""
import time, os, tempfile, shutil

def bench(fn, n=500, warm=10):
    for _ in range(warm):
        try: fn()
        except: pass
    t0 = time.perf_counter()
    for _ in range(n):
        try: fn()
        except: pass
    t1 = time.perf_counter()
    return (t1 - t0) / n * 1000

results = []

# ── 1. Category.is_primitive ─────────────────────
from fastnltk._rust import from_string, Category
c = from_string("NP")
ms = bench(lambda: c.is_primitive(), 10000)
results.append(("ccg/Category.is_primitive", ms, "—"))

# NLTK comparison
from nltk.ccg.api import PrimitiveCategory as NLTK_PC
nc = NLTK_PC("NP")
nl = bench(lambda: nc.is_primitive(), 5000)
results.append(("ccg/NLTK Category.is_primitive", nl, f"{nl/ms:.1f}x" if ms else "—"))

# ── 2. StupidBackoff.fit ────────────────────────
from fastnltk._rust import StupidBackoff
train = [["the", "cat", "sat"], ["the", "dog", "ran"], ["the", "cat", "ran"]]
ms = bench(lambda: StupidBackoff(3).fit(train), 200)
results.append(("lm/StupidBackoff.fit", ms, "—"))

from nltk.lm import StupidBackoff as NLTK_SB
from nltk.lm.preprocessing import padded_everygram_pipeline
nltk_train = [["the", "cat", "sat"], ["the", "dog", "ran"]]
nl = bench(lambda: NLTK_SB(order=3).fit(padded_everygram_pipeline(3, nltk_train)[0]), 20)
results.append(("lm/NLTK StupidBackoff.fit", nl, f"{nl/ms:.1f}x" if ms else "—"))

# ── 3. BigramAssocMeasures.pmi ──────────────────
from fastnltk._rust import BigramAssocMeasures
ms = bench(lambda: BigramAssocMeasures.pmi(10, 100, 50, 1000), 20000)
results.append(("metrics/BigramAssocMeasures.pmi", ms, "—"))

from nltk.metrics.association import BigramAssocMeasures as NLTK_BAM
nl = bench(lambda: NLTK_BAM.pmi(10, 100, 50, 1000), 5000)
results.append(("metrics/NLTK BigramAssocMeasures.pmi", nl, f"{nl/ms:.1f}x" if ms else "—"))

# ── 4. ConditionalFreqDist.conditions ────────────
from fastnltk._rust import ConditionalFreqDist
cfd = ConditionalFreqDist()
cfd.inc("news", "word")
cfd.inc("sports", "ball")
ms = bench(lambda: cfd.conditions(), 10000)
results.append(("probability/ConditionalFreqDist.conditions", ms, "—"))

from nltk.probability import ConditionalFreqDist as NLTK_CFD
ncfd = NLTK_CFD()
ncfd["news"]["word"] += 2
ncfd["sports"]["ball"] += 1
nl = bench(lambda: ncfd.conditions(), 2000)
results.append(("probability/NLTK CFD.conditions", nl, f"{nl/ms:.1f}x" if ms else "—"))

# ── 5. UnigramTagger.train ───────────────────────
from fastnltk._rust import UnigramTagger as Rust_UT
train_sents = [[("the", "DT"), ("cat", "NN"), ("sat", "VBD")]]
ms = bench(lambda: Rust_UT().train(train_sents), 500)
results.append(("tag/UnigramTagger.train", ms, "—"))

from nltk.tag import UnigramTagger as NLTK_UT
nl = bench(lambda: NLTK_UT.train(train_sents), 200)
results.append(("tag/NLTK UnigramTagger.train", nl, f"{nl/ms:.1f}x" if ms else "—"))

# ── 6. StupidBackoff.score ──────────────────────
# (score is already in benchmark table, fit is not)
# Let's also bench score with a fitted model
sb = StupidBackoff(3)
sb.fit([["the", "cat", "sat"]])
ms = bench(lambda: sb.score("cat", ["the"]), 5000)
results.append(("lm/StupidBackoff.score (warm)", ms, "—"))

# ── Print ────────────────────────────────────────
print(f"{'Function':50s} {'Time':>10s} {'Speedup':>8s}")
print("="*68)
for name, t, speed in results:
    unit = "ms" if t >= 1 else "us"
    val = t if t >= 1 else t * 1000
    print(f"{name:50s} {val:>7.2f}{unit}  {speed:>6s}")
