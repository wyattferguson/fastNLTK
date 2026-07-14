"""Benchmarks for new Phase A+B+C tokenizers and metrics."""
import time, statistics, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
def read_text(size):
    with open(os.path.join(DATA_DIR, f"{size}.txt")) as f:
        return f.read()
def bench(name, fn, *args, iterations=50):
    for _ in range(3): fn(*args)
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    mean = statistics.mean(times)
    sd = statistics.stdev(times) if len(times) > 1 else 0
    print(f"  {name}: {mean:.2f}ms (+-{sd:.2f}ms)")
    return mean

def main():
    import nltk, fastnltk
    results = []

    # SExpr — use balanced pattern "(a)" which is always properly nested
    print("\n=== SExprTokenizer ===")
    sexpr_text = "(a) " * 2000  # 8000 chars, balanced
    for size, txt in [("small", sexpr_text[:2000]), ("medium", sexpr_text)]:
        print(f"  Input: {size} ({len(txt)} chars)")
        if size == "small":
            t_n = bench("NLTK", nltk.tokenize.SExprTokenizer().tokenize, txt, iterations=50)
            t_f = bench("fastNLTK", fastnltk.tokenize.SExprTokenizer().tokenize, txt, iterations=50)
        else:
            t_n = bench("NLTK", nltk.tokenize.SExprTokenizer().tokenize, txt, iterations=20)
            t_f = bench("fastNLTK", fastnltk.tokenize.SExprTokenizer().tokenize, txt, iterations=20)
        sp = t_n / t_f if t_f > 0 else 0
        print(f"  Speedup: {sp:.1f}x")
        results.append(("SExprTokenizer.tokenize", size, len(txt), t_n, t_f, sp))

    # TokTok
    print("\n=== ToktokTokenizer ===")
    text_med = read_text("medium")
    for size, txt in [("small", text_med[:5000]), ("medium", text_med)]:
        print(f"  Input: {size} ({len(txt)} chars)")
        t_n = bench("NLTK", nltk.tokenize.ToktokTokenizer().tokenize, txt, iterations=20)
        t_f = bench("fastNLTK", fastnltk.tokenize.ToktokTokenizer().tokenize, txt, iterations=20)
        sp = t_n / t_f if t_f > 0 else 0
        print(f"  Speedup: {sp:.1f}x")
        results.append(("ToktokTokenizer.tokenize", size, len(txt), t_n, t_f, sp))

    # MWE
    print("\n=== MWETokenizer ===")
    mwes = [("New", "York"), ("Los", "Angeles"), ("San", "Francisco")]
    words = ("New York is a city. Los Angeles is bigger. San Francisco is nice. " * 2000).split()
    print(f"  Input: {len(words)} words")
    t_n = bench("NLTK", nltk.tokenize.MWETokenizer(mwes).tokenize, words, iterations=20)
    t_f = bench("fastNLTK", fastnltk.tokenize.MWETokenizer(mwes, "_").tokenize, words, iterations=20)
    sp = t_n / t_f if t_f > 0 else 0
    results.append(("MWETokenizer.tokenize", "10K words", len(words), t_n, t_f, sp))
    print(f"  Speedup: {sp:.1f}x")

    # Segmentation
    print("\n=== windowdiff ===")
    s1 = "000100000010" * 1000
    s2 = "000010000100" * 1000
    print(f"  Input: {len(s1)} chars")
    t_n = bench("NLTK", nltk.metrics.segmentation.windowdiff, s1, s2, 3, "1", False, iterations=100)
    t_f = bench("fastNLTK", fastnltk.metrics.windowdiff, s1, s2, 3, "1", iterations=100)
    sp = t_n / t_f if t_f > 0 else 0
    print(f"  Speedup: {sp:.1f}x")
    results.append(("windowdiff", "10K chars", len(s1), t_n, t_f, sp))

    # KneserNey
    print("\n=== KneserNeyInterpolated ===")
    train_data = [["the", "cat", "sat"], ["the", "dog", "ran"], ["a", "cat", "ran"]]
    text = ["the", "cat", "ran", "a", "dog", "sat"]
    kn_n = nltk.lm.KneserNeyInterpolated(2)
    kn_f = fastnltk.lm.KneserNeyInterpolated(2, 0.75)
    def run_nltk_kn():
        kn_n.fit(train_data); return [kn_n.score(w, ["the"]) for w in text]
    def run_fast_kn():
        kn_f.fit(train_data); return [kn_f.score(w, ["the"]) for w in text]
    t_n = bench("NLTK", run_nltk_kn, iterations=20)
    t_f = bench("fastNLTK", run_fast_kn, iterations=20)
    sp = t_n / t_f if t_f > 0 else 0
    print(f"  Speedup: {sp:.1f}x")
    results.append(("KneserNeyInterpolated", "small", 50, t_n, t_f, sp))

    # TextTiling
    print("\n=== TextTilingTokenizer ===")
    tt_n = nltk.tokenize.TextTilingTokenizer(w=20, k=10, demo_mode=True)
    tt_f = fastnltk.tokenize.TextTilingTokenizer(20, 10, True)
    text = read_text("medium")
    print(f"  Input: {len(text)} chars")
    t_n = bench("NLTK", lambda: tt_n.tokenize(text), iterations=10)
    t_f = bench("fastNLTK", lambda: tt_f.tokenize(text), iterations=10)
    sp = t_n / t_f if t_f > 0 else 0
    print(f"  Speedup: {sp:.1f}x")
    results.append(("TextTilingTokenizer.tokenize", "medium", len(text), t_n, t_f, sp))

    # Save
    import json
    out = {"benchmarks": []}
    for name, size, n_chars, nltk_ms, fast_ms, speedup in results:
        out["benchmarks"].append({
            "name": name, "params": {"size": size, "chars": n_chars},
            "stats": {"mean": nltk_ms / 1000, "mean_fast": fast_ms / 1000},
            "speedup": speedup,
        })
    out_path = os.path.join(os.path.dirname(__file__), "results", "phase_abc_bench.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
