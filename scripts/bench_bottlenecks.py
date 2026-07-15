"""Benchmark: measure bottlenecks in fastNLTK hot paths."""
import time
from fastnltk._rust import TreebankWordTokenizer, RegexpTokenizer
from fastnltk import pos_tag

# --- data ---
text = " ".join(["word" + str(i) for i in range(10000)])
text_long = " ".join(["The quick brown fox jumps over the lazy dog ." for _ in range(2000)])

# 1. Throughput comparison
t_re = RegexpTokenizer(r"\S+", gaps=False)
t_tb = TreebankWordTokenizer()

t0 = time.perf_counter()
for _ in range(200):
    r = text.split()
t1 = time.perf_counter()
split_ms = (t1 - t0) / 200 * 1000

t0 = time.perf_counter()
for _ in range(100):
    r = t_re.tokenize(text)
t1 = time.perf_counter()
re_ms = (t1 - t0) / 100 * 1000

t0 = time.perf_counter()
for _ in range(100):
    r = t_tb.tokenize(text)
t1 = time.perf_counter()
tb_ms = (t1 - t0) / 100 * 1000

print(f"str.split:            {split_ms:.3f} ms  ({len(text.split())} tok)")
print(f"Regexp \\S+:           {re_ms:.3f} ms  ({len(r)} tok)  ({re_ms/split_ms:.1f}x)")
print(f"Treebank:             {tb_ms:.3f} ms  ({len(r)} tok)  ({tb_ms/split_ms:.1f}x)")

# 2. pos_tag (real-world usage)
tokens = text_long.split()[:1000]
t0 = time.perf_counter()
for _ in range(50):
    r = pos_tag(tokens)
t1 = time.perf_counter()
print(f"\npos_tag (1000 words):  {(t1-t0)/50*1000:.2f} ms  ({len(r)} tags)")

# 3. What would we gain from SIMD / manual scanner?
# str.split on 10K words: ~0.2ms. Regex \S+: ~0.9ms. The regex engine
# is doing DFA construction, UTF-8 decoding, capture groups.
# A hand-written SIMD whitespace scanner would be 5-10x faster than regex.
print(f"\nRegex overhead:        {(re_ms/split_ms - 1)*100:.0f}% slower than str.split")
print(f"Theoretical gain:      ~5-10x with SIMD whitespace scanner")
